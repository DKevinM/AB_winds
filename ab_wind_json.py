# ab_wind_json.py
import json
import os
import tempfile
import requests
import xarray as xr



HRDPS_BASE = "https://dd.weather.gc.ca/model_hrdps/continental/2.5km/grib2"


# Rough Alberta bounding box
AB_LAT_MIN, AB_LAT_MAX = 48.5, 60.5
AB_LON_MIN, AB_LON_MAX = -120.0, -108.0  # adjust if you want more margin



def download_to_temp(url: str) -> str:
    """
    Download a GRIB2 file from 'url' into a temporary file.
    Returns the file path. Caller is responsible for os.remove(path).
    """
    print(f"Downloading: {url}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    fd, path = tempfile.mkstemp(suffix=".grib2")
    with os.fdopen(fd, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"Saved GRIB2 to temp file: {path}")
    return path


def open_hrdps_10m_uv(path_u: str, path_v: str):
    """
    Open HRDPS 10 m U/V GRIB2 files and return (u10, v10) as xarray DataArrays.

    Assumes:
    - TypeOfLevel heightAboveGround
    - level = 10 m
    """
    filter_kwargs = {
        "filter_by_keys": {
            "typeOfLevel": "heightAboveGround",
            "level": 10
        }
    }

    ds_u = xr.open_dataset(path_u, engine="cfgrib", backend_kwargs=filter_kwargs)
    ds_v = xr.open_dataset(path_v, engine="cfgrib", backend_kwargs=filter_kwargs)

    # Be robust to variable names
    def pick_var(ds, candidates):
        for name in candidates:
            if name in ds:
                return ds[name]
        raise ValueError(f"None of {candidates} found in dataset variables: {list(ds.data_vars)}")

    u = pick_var(ds_u, ["u10", "10u", "UGRD"])
    v = pick_var(ds_v, ["v10", "10v", "VGRD"])

    # Subset to Alberta bounding box
    # HRDPS longitudes are usually 0–360; convert our mins/max to that range
    lon_min = AB_LON_MIN % 360
    lon_max = AB_LON_MAX % 360

    sub_u = u.sel(
        latitude=slice(AB_LAT_MAX, AB_LAT_MIN),  # lat typically descending
        longitude=slice(lon_min, lon_max)
    )
    sub_v = v.sel(
        latitude=slice(AB_LAT_MAX, AB_LAT_MIN),
        longitude=slice(lon_min, lon_max)
    )

    # Use the first time slice (extend later if you want multiple times)
    if "time" in sub_u.dims:
        sub_u = sub_u.isel(time=0)
    if "time" in sub_v.dims:
        sub_v = sub_v.isel(time=0)

    return sub_u, sub_v


def to_earth_like_json(u, v):
    """
    Convert u/v DataArrays into a compact JSON structure the JS side can read.
    """
    # Ensure shapes match
    if u.shape != v.shape:
        raise ValueError(f"u and v shapes differ: {u.shape} vs {v.shape}")

    lats = u["latitude"].values
    lons = u["longitude"].values

    meta = {
        "lat_min": float(lats.min()),
        "lat_max": float(lats.max()),
        "lon_min": float(lons.min()),
        "lon_max": float(lons.max()),
        "nlat": int(len(lats)),
        "nlon": int(len(lons)),
        "time": str(u.coords.get("time").values) if "time" in u.coords else None,
        "units": {
            "u": "m/s",
            "v": "m/s"
        }
    }

    # Convert to native Python lists (float32 to keep reasonably small)
    u_vals = u.values.astype("float32").tolist()
    v_vals = v.values.astype("float32").tolist()

    return {
        "meta": meta,
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "u": u_vals,
        "v": v_vals
    }


def build_hrdps_url(run_time, var, lead_hour):
    """
    var        = 'UGRD' or 'VGRD'
    lead_hour  = integer forecast hour (1, 2, 3...)
    """
    cycle = f"{run_time.hour:02d}"          # 00, 06, 12, 18
    date_str = run_time.strftime("%Y%m%d")   # 20251212

    # HRDPS file naming pattern:
    # {DATE}{CYCLE}_MSC_HRDPS_{VAR}_AGL-10m_RLatLon0.0225_PT{HOUR}H.grib2

    filename = (
        f"{date_str}{cycle}_MSC_HRDPS_{var}_AGL-10m_RLatLon0.0225_PT{lead_hour:03d}H.grib2"
    )

    # HRDPS directory structure:
    # model_hrdps/.../grib2/{cycle}/{lead_hour}/{filename}
    url = f"{HRDPS_BASE}/{cycle}/{lead_hour:03d}/{filename}"
    return url





def main():
    # For now, point to explicit HRDPS UGRD/VGRD URLs.
    # Later we can auto-build these based on current UTC + run hour.
    #
    # You can also pass them in via environment variables to keep YAML cleaner:
    #   HRDPS_, HRDPS_VGRD_URL
    run_time = pick_run_cycle()
    
    ugrd_url = build_hrdps_url(run_time, "UGRD", lead_hour=1)
    vgrd_url = build_hrdps_url(run_time, "VGRD", lead_hour=1)
    
    print("UGRD URL:", ugrd_url)
    print("VGRD URL:", vgrd_url)

    if not ugrd_url or not vgrd_url:
        raise SystemExit("HRDPS_UGRD_URL and HRDPS_VGRD_URL environment variables must be set")

    u_path = v_path = None
    try:
        u_path = download_to_temp(ugrd_url)
        v_path = download_to_temp(vgrd_url)

        u10, v10 = open_hrdps_10m_uv(u_path, v_path)
        js = to_earth_like_json(u10, v10)

        out_dir = "data"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "AB_wind_latest.json")

        with open(out_path, "w") as f:
            json.dump(js, f)

        print(f"Wrote {out_path}")

    finally:
        # Clean up temp files
        if u_path and os.path.exists(u_path):
            os.remove(u_path)
        if v_path and os.path.exists(v_path):
            os.remove(v_path)


if __name__ == "__main__":
    main()
