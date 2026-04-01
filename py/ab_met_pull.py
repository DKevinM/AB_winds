# ab_met_pull.py

import os, json, gzip, tempfile, requests, datetime as dt
import numpy as np
import xarray as xr

from supabase import create_client


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
AB_LON_MIN, AB_LON_MAX = -125, -100

LEVELS_AGL = [10, 40, 80, 120]

FIELDS = {
    "UGRD": [f"AGL-{h}m" for h in LEVELS_AGL],
    "VGRD": [f"AGL-{h}m" for h in LEVELS_AGL],
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

def crop(ds):
    lon = ds.longitude % 360
    mask = (ds.latitude >= AB_LAT_MIN) & (ds.latitude <= AB_LAT_MAX) & \
           (lon >= AB_LON_MIN % 360) & (lon <= AB_LON_MAX % 360)
    return ds.where(mask, drop=True)



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

                ds = xr.open_dataset(tmp, engine="cfgrib")
                sub = crop(ds)

                lats = sub.latitude.values
                lons = sub.longitude.values
                lons = np.where(lons > 180, lons - 360, lons)

                ny, nx = lats.shape if lats.ndim == 2 else (len(lats), len(lons))

                lat_max = float(np.nanmax(lats))
                lon_min = float(np.nanmin(lons))

                dx = float((np.nanmax(lons) - np.nanmin(lons)) / (nx - 1))
                dy = float((lat_max - float(np.nanmin(lats))) / (ny - 1))

                grid = {
                    "lo1": lon_min,
                    "la1": lat_max,
                    "dx": dx,
                    "dy": dy,
                    "nx": nx,
                    "ny": ny
                }

                key = f"{var.lower()}{lvl.replace('AGL-','').replace('m','')}"
                fields_out[key] = sub.to_array().values.astype("float32").tolist()

                os.remove(tmp)

        if not any_success:
            print(f"Stopping at lead {lead} — no data available")
            break

        payload = {
            "meta": {
                "run": run.isoformat()+"Z",
                "lead": lead,
                "valid": (run + dt.timedelta(hours=lead)).isoformat()+"Z"
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

        


        # -------------------------------
        # Upload to Supabase (NEW STEP)
        # -------------------------------
        
        filename = os.path.basename(fname)
        
        # extract date for storage path
        year  = int(run.strftime("%Y"))
        month = int(run.strftime("%m"))
        day   = int(run.strftime("%d"))
        
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
