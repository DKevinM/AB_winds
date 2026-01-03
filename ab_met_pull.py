# ab_met_pull.py

import os, json, gzip, tempfile, requests, datetime as dt
import numpy as np
import xarray as xr

# ================= CONFIG =================
BASE = "https://dd.weather.gc.ca/today/model_hrdps/continental/2.5km"

AB_LAT_MIN, AB_LAT_MAX = 48.5, 60.5
AB_LON_MIN, AB_LON_MAX = -120, -108

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

def pick_cycle():
    now = dt.datetime.utcnow()
    return now.replace(hour=(now.hour//6)*6, minute=0, second=0, microsecond=0)

def build_url(run, var, level, lead):
    tag = run.strftime("%Y%m%dT%HZ")
    return f"{BASE}/{run:%H}/{lead:03d}/{tag}_MSC_HRDPS_{var}_{level}_RLatLon0.0225_PT{lead:03d}H.grib2"

def download(url):
    r = requests.get(url)
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

def main():
    run = pick_cycle()
    os.makedirs(OUTDIR, exist_ok=True)

    for lead in LEADS:
        fields_out = {}

        for var, levels in FIELDS.items():
            for lvl in levels:
                url = build_url(run, var, lvl, lead)
                print("Downloading", url)

                tmp = download(url)
                ds = xr.open_dataset(tmp, engine="cfgrib")
                sub = crop(ds)

                key = f"{var.lower()}{lvl.replace('AGL-','').replace('m','')}"
                fields_out[key] = sub.to_array().values.astype("float32").tolist()

                os.remove(tmp)

        payload = {
            "meta": {
                "run": run.isoformat()+"Z",
                "lead": lead,
                "valid": (run + dt.timedelta(hours=lead)).isoformat()+"Z"
            },
            "fields": fields_out
        }

        fname = f"{OUTDIR}/ab_met_{run:%Y%m%d_%HZ}_f{lead:03d}.json.gz"
        with gzip.open(fname, "wt", encoding="utf-8") as f:
            json.dump(payload, f)

        print("Saved:", fname)

if __name__ == "__main__":
    main()
