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
    
def pick_available_cycle(now=None):
    if now is None:
        now = dt.datetime.utcnow()

    for h in range(0, 49, 3):
        test_time = now - dt.timedelta(hours=h)
        cycle_hour = (test_time.hour // 6) * 6
        cycle = test_time.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

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
    run = pick_available_cycle()
    print("Using HRDPS cycle:", run.isoformat())
    
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
        with gzip.open(fname, "wt", encoding="utf-8") as f:
            json.dump(payload, f)

        print("Saved:", fname)

if __name__ == "__main__":
    main()
