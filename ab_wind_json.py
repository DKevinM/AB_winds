# ab_wind_json.py
import json
import os
import tempfile
import requests
import xarray as xr
import datetime as dt

# "today" HRDPS base: matches URLs like
# https://dd.weather.gc.ca/today/model_hrdps/continental/2.5km/12/001/20251212T12Z_MSC_HRDPS_UGRD_AGL-10m_RLatLon0.0225_PT001H.grib2
HRDPS_BASE = "https://dd.weather.gc.ca/today/model_hrdps/continental/2.5km"

# Rough Alberta bounding box
AB_LAT_MIN, AB_LAT_MAX = 48.5, 60.5
AB_LON_MIN, AB_LON_MAX = -120.0, -108.0  # adjust if you want more margin


def download_to_temp(url: str) -> str:
    """
    Download a GRIB2 file from 'url' into a temporary file.
    Returns the file path. Caller is responsible for os.remove(path).
    """
    print(f"Downloading: {url}")
    resp = requests.get(url, stream=True, timeout=60)
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
    Subsets to Alberta bounding box.

    Works whether lat/lon are 1-D dims (latitude, longitude) or 2-D coords (latitude(y,x), longitude(y,x)).
    """
    filter_kwargs = {
        "filter_by_keys": {
            "typeOfLevel": "heightAboveGround",
            "level": 10
        }
    }

    ds_u = xr.open_dataset(path_u, engine="cfgrib", backend_kwargs=filter_kwargs)
    ds_v = xr.open_dataset(path_v, engine="cfgrib", backend_kwargs=filter_kwargs)

    def pick_var(ds, candidates):
        for name in candidates:
            if name in ds:
                return ds[name]
        raise ValueError(f"None of {candidates} found in dataset variables: {list(ds.data_vars)}")

    u = pick_var(ds_u, ["u10", "10u", "UGRD"])
    v = pick_var(ds_v, ["v10", "10v", "VGRD"])

    # take first time slice if present
    if "time" in u.dims:
        u = u.isel(time=0)
    if "time" in v.dims:
        v = v.isel(time=0)

    # longitudes in HRDPS are often 0..360
    lon_min = AB_LON_MIN % 360
    lon_max = AB_LON_MAX % 360

    # CASE A: 1-D lat/lon dims exist -> fast .sel slicing
    # (works if 'latitude' and 'longitude' are dimension coords)
    if ("latitude" in u.dims) and ("longitude" in u.dims):
        sub_u = u.sel(latitude=slice(AB_LAT_MAX, AB_LAT_MIN),
                      longitude=slice(lon_min, lon_max))
        sub_v = v.sel(latitude=slice(AB_LAT_MAX, AB_LAT_MIN),
                      longitude=slice(lon_min, lon_max))
        return sub_u, sub_v

    # CASE B: lat/lon are 2-D coords on (y,x) -> mask + drop
    latc = u.coords.get("latitude", None)
    lonc = u.coords.get("longitude", None)

    if latc is None or lonc is None:
        raise ValueError(f"Missing latitude/longitude coords. u coords: {list(u.coords)}")

    lonc_360 = (lonc % 360)

    mask = (
        (latc >= AB_LAT_MIN) & (latc <= AB_LAT_MAX) &
        (lonc_360 >= lon_min) & (lonc_360 <= lon_max)
    )

    sub_u = u.where(mask, drop=True)
    sub_v = v.where(mask, drop=True)

    return sub_u, sub_v



def to_earth_like_json(u, v):
    if u.shape != v.shape:
        raise ValueError(f"u and v shapes differ: {u.shape} vs {v.shape}")

    # Dimensions (prefer y/x if present)
    if "y" in u.dims and "x" in u.dims:
        ny = int(u.sizes["y"])
        nx = int(u.sizes["x"])
        y = u["y"].values.tolist()
        x = u["x"].values.tolist()

        lat2d = u.coords.get("latitude")
        lon2d = u.coords.get("longitude")
        if lat2d is None or lon2d is None:
            raise ValueError(f"Expected 2D latitude/longitude coords. Coords: {list(u.coords)}")

        lat_vals = lat2d.values.astype("float32")
        lon_vals = (lon2d.values % 360).astype("float32")

        meta = {
            "grid": "yx_with_latlon2d",
            "ny": ny,
            "nx": nx,
            "lat_min": float(lat_vals.min()),
            "lat_max": float(lat_vals.max()),
            "lon_min": float(lon_vals.min()),
            "lon_max": float(lon_vals.max()),
            "time": str(u.coords.get("time").values) if "time" in u.coords else None,
            "units": {"u": "m/s", "v": "m/s"},
        }

        return {
            "meta": meta,
            "y": y,
            "x": x,
            "lat2d": lat_vals.tolist(),
            "lon2d": lon_vals.tolist(),
            "u": u.values.astype("float32").tolist(),  # shape [ny][nx]
            "v": v.values.astype("float32").tolist(),
        }

    # Rectilinear case: 1-D latitude/longitude dims
    if ("latitude" in u.dims) and ("longitude" in u.dims):
        lats = u["latitude"].values
        lons = (u["longitude"].values % 360)

        meta = {
            "grid": "latlon1d",
            "lat_min": float(lats.min()),
            "lat_max": float(lats.max()),
            "lon_min": float(lons.min()),
            "lon_max": float(lons.max()),
            "nlat": int(len(lats)),
            "nlon": int(len(lons)),
            "time": str(u.coords.get("time").values) if "time" in u.coords else None,
            "units": {"u": "m/s", "v": "m/s"},
        }

        return {
            "meta": meta,
            "lats": lats.tolist(),
            "lons": lons.tolist(),
            "u": u.values.astype("float32").tolist(),  # [nlat][nlon]
            "v": v.values.astype("float32").tolist(),
        }

    raise ValueError(f"Unsupported grid/dims. dims={u.dims}, coords={list(u.coords)}")





def pick_run_cycle(now: dt.datetime | None = None) -> dt.datetime:
    """
    Pick the most recent HRDPS cycle time (00, 06, 12, 18 UTC)
    based on current UTC time.
    """
    if now is None:
        now = dt.datetime.utcnow()

    hour = now.hour
    if hour < 6:
        cycle_hour = 0
    elif hour < 12:
        cycle_hour = 6
    elif hour < 18:
        cycle_hour = 12
    else:
        cycle_hour = 18

    return now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)





def build_hrdps_url(run_time: dt.datetime, var: str, lead_hour: int) -> str:
    """
    Build an HRDPS URL for a given run time, variable and forecast hour.

    - var: 'UGRD' or 'VGRD'
    - lead_hour: 0, 1, 2, 3, ...
    """
    date_str = run_time.strftime("%Y%m%d")  # 20251212

    # '00', '06', '12', '18'
    cycle_hour = run_time.hour
    cycle_dir = f"{cycle_hour:02d}"

    # '00Z', '06Z', ...
    cycle_tag = f"{cycle_dir}Z"

    # '000', '001', ...
    lead_str = f"{lead_hour:03d}"

    filename = (
        f"{date_str}T{cycle_tag}_MSC_HRDPS_{var}_AGL-10m_RLatLon0.0225_PT{lead_str}H.grib2"
    )

    # today/model_hrdps/.../{cycle}/{lead}/{filename}
    url = f"{HRDPS_BASE}/{cycle_dir}/{lead_str}/{filename}"
    return url


def main():
    run_time = pick_run_cycle()
    for attempt in range(2):  # current cycle, then previous cycle
        test_url = build_hrdps_url(run_time, "UGRD", 0)
        r = requests.head(test_url)
        if r.status_code == 200:
            break
        run_time = run_time - dt.timedelta(hours=6)
    
    print("Using HRDPS run cycle:", run_time.isoformat())


    # Forecast hours you care about
    lead_hours = [0, 1, 2, 3]

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    for lead_hour in lead_hours:
        print(f"Processing lead hour {lead_hour}h")

        ugrd_url = build_hrdps_url(run_time, "UGRD", lead_hour)
        vgrd_url = build_hrdps_url(run_time, "VGRD", lead_hour)

        print("UGRD URL:", ugrd_url)
        print("VGRD URL:", vgrd_url)

        u_path = v_path = None
        try:
            u_path = download_to_temp(ugrd_url)
            v_path = download_to_temp(vgrd_url)

            u10, v10 = open_hrdps_10m_uv(u_path, v_path)
            js = to_earth_like_json(u10, v10)

            out_path = os.path.join(out_dir, f"AB_wind_{lead_hour:03d}.json")
            with open(out_path, "w") as f:
                json.dump(js, f)
            print(f"Wrote {out_path}")

            # Also keep AB_wind_latest.json as the 0-hour field for compatibility
            if lead_hour == 0:
                latest_path = os.path.join(out_dir, "AB_wind_latest.json")
                with open(latest_path, "w") as f:
                    json.dump(js, f)
                print(f"Wrote {latest_path}")

        finally:
            if u_path and os.path.exists(u_path):
                os.remove(u_path)
            if v_path and os.path.exists(v_path):
                os.remove(v_path)


if __name__ == "__main__":
    main()
