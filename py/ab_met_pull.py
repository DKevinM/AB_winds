# ab_met_pull.py

import os, json, gzip, tempfile, requests, datetime as dt
import numpy as np
import xarray as xr

from supabase import create_client

from hrdps_regrid import regridder_for, GRID_VERSION


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def wind_file_exists(file_path):

    resp = supabase.table("wind_files") \
        .select("id") \
        .eq("file_path", file_path) \
        .limit(1) \
        .execute()

    return len(resp.data) > 0


def main(backfill_hours=0):
    now = dt.datetime.utcnow()
    runs_to_process = []

    # current run
    try:
        runs_to_process.append(pick_available_cycle(now))
    except:
        print("No current cycle found")

    # backfill runs
    for h in range(6, backfill_hours + 1, 6):
        past_time = now - dt.timedelta(hours=h)
        try:
            runs_to_process.append(pick_available_cycle(past_time))
        except:
            continue

    # remove duplicates
    runs_to_process = list({r: None for r in runs_to_process}.keys())
    print("Runs to process:", runs_to_process)
    for run in runs_to_process:
        process_run(run)


def upload_to_supabase(local_path, storage_path):

    with open(local_path, "rb") as f:
        try:
            supabase.storage.from_("winds").upload(
                path=storage_path,
                file=f,
                file_options={
                    "content-type": "application/json",
                    "content-encoding": "gzip"
                }
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                print("File already exists in storage")
            else:
                raise



import re
from datetime import datetime, timedelta

def insert_metadata_from_run(run, lead, storage_path):

    run_time = run
    valid_time = run + timedelta(hours=lead)

    supabase.table("wind_files").insert({
        "model": "HRDPS",
        "run_time": run_time.isoformat(),
        "forecast_hour": lead,
        "valid_time": valid_time.isoformat(),

        "year": valid_time.year,
        "month": valid_time.month,
        "day": valid_time.day,
        "hour": valid_time.hour,

        "file_path": storage_path,
        "file_format": "json.gz"
    }).execute()




# ================= CONFIG =================
BASE = "https://dd.weather.gc.ca/today/model_hrdps/continental/2.5km"

AB_LAT_MIN, AB_LAT_MAX = 48, 62
AB_LON_MIN, AB_LON_MAX = -125, -88

LEVELS_AGL = [10, 40, 80, 120]
# HRDPS doesn't publish TMP at 10m (that height is wind-only; temperature's
# surface level is 2m) - confirmed live against dd.weather.gc.ca before adding.
TMP_LEVELS_AGL = [2, 40, 80, 120]

FIELDS = {
    "UGRD": [f"AGL-{h}m" for h in LEVELS_AGL],
    "VGRD": [f"AGL-{h}m" for h in LEVELS_AGL],
    "TMP": [f"AGL-{h}m" for h in TMP_LEVELS_AGL],
    "HPBL": ["Sfc"],
    "RH": ["AGL-2m"],
    "DPT": ["AGL-2m"]
}

LEADS = [0,3,6,9,12,15,18,21,24]
OUTDIR = "met_data"

# ========================================
    
def pick_available_cycle(now=None):
    if now is None:
        now = dt.datetime.utcnow()

    for h in range(0, 49, 3):
        test_time = now - dt.timedelta(hours=h)

        cycle_hour = (test_time.hour // 6) * 6
        cycle = test_time.replace(
            hour=cycle_hour,
            minute=0,
            second=0,
            microsecond=0
        )

        cycle_dir = f"{BASE}/{cycle:%H}/000/"

        try:
            html = requests.get(cycle_dir, timeout=15).text
        except Exception:
            continue

        if "UGRD_AGL-10m" in html:
            print("Found available HRDPS cycle:", cycle.isoformat())
            return cycle

    raise RuntimeError("No available HRDPS cycle found")


    

def build_url(run, var, level, lead):
    tag = run.strftime("%Y%m%dT%HZ")
    return f"{BASE}/{run:%H}/{lead:03d}/{tag}_MSC_HRDPS_{var}_{level}_RLatLon0.0225_PT{lead:03d}H.grib2"

def download(url):
    r = requests.get(url)
    if r.status_code == 404:
        print(f" Missing file (skipping): {url}")
        return None
    r.raise_for_status()
    
    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(r.content)
    f.close()
    return f.name

def process_run(run):

    print("Processing run:", run.isoformat())

    os.makedirs(OUTDIR, exist_ok=True)

    for lead in LEADS:

        fields_out = {}
        any_success = False

        for var, levels in FIELDS.items():
            for lvl in levels:

                url = build_url(run, var, lvl, lead)
                print("Downloading", url)

                tmp = download(url)

                if tmp is None:
                    continue

                any_success = True

                # HRDPS is on a rotated grid: resample onto the regular north-up
                # lat/lon grid every reader assumes (see hrdps_regrid.py).
                rg = regridder_for(tmp)
                ds = xr.open_dataset(tmp, engine="cfgrib")
                arr = ds.to_array().values[0]
                ds.close()

                grid = rg.grid
                key = f"{var.lower()}{lvl.replace('AGL-','').replace('m','')}"
                fields_out[key] = rg.resample(arr)

                os.remove(tmp)

        # UGRD/VGRD are relative to the rotated grid axes; turn them to true east/north.
        if fields_out:
            for h in LEVELS_AGL:
                uk, vk = f"ugrd{h}", f"vgrd{h}"
                if uk in fields_out and vk in fields_out:
                    fields_out[uk], fields_out[vk] = rg.earth_relative(fields_out[uk], fields_out[vk])
                elif uk in fields_out or vk in fields_out:
                    # an unrotated lone component would be silently wrong
                    fields_out.pop(uk, None)
                    fields_out.pop(vk, None)
                    print(f" Dropped level {h} m: only one wind component downloaded")
            # 0.01 precision (m/s, K, %, m) is well inside model accuracy and keeps
            # the JSON compact; float64 so values serialize as e.g. 1.23, not 1.2300000190734863
            fields_out = {k: [np.round(a.astype(np.float64), 2).tolist()] for k, a in fields_out.items()}

        if not any_success:
            print(f"Stopping at lead {lead} — no data available")
            break

        payload = {
            "meta": {
                "run": run.isoformat()+"Z",
                "lead": lead,
                "valid": (run + dt.timedelta(hours=lead)).isoformat()+"Z",
                "grid_version": GRID_VERSION,
                "winds": "earth-relative (u east, v north), m/s"
            },
            "grid": grid,
            "fields": fields_out
        }

        fname = f"{OUTDIR}/ab_met_{run:%Y%m%d_%HZ}_f{lead:03d}.json.gz"

        def sanitize_json(obj):
            if isinstance(obj, dict):
                return {k: sanitize_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize_json(v) for v in obj]
            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
            return obj
        
        clean_payload = sanitize_json(payload)
        
        with gzip.open(fname, "wt", encoding="utf-8") as f:
            json.dump(clean_payload, f, allow_nan=False)

        print("Saved:", fname)

        filename = os.path.basename(fname)

        valid_time = run + dt.timedelta(hours=lead)
        storage_path = f"hrdps/{valid_time.year}/{valid_time.month:02d}/{valid_time.day:02d}/{filename}"

        if not wind_file_exists(storage_path):

            print("Uploading to Supabase:", filename)

            try:
                upload_to_supabase(fname, storage_path)
                insert_metadata_from_run(run, lead, storage_path)

            except Exception as e:
                if "already exists" in str(e):
                    print("Skipped existing file")
                else:
                    raise

        else:
            print("Already exists in Supabase:", filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", type=int, default=0)

    args = parser.parse_args()

    main(backfill_hours=args.backfill)
