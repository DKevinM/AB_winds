# ab_wind_json.py
import json
import os
import tempfile
import requests
import xarray as xr
import datetime as dt
import numpy as np

HRDPS_BASE = "https://dd.weather.gc.ca/today/model_hrdps/continental/2.5km"

AB_LAT_MIN, AB_LAT_MAX = 48.5, 60.5
AB_LON_MIN, AB_LON_MAX = -120.0, -108.0  # degrees East negative for W

def download_to_temp(url: str) -> str:
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

    # HRDPS longitudes often 0..360
    lon_min_360 = AB_LON_MIN % 360
    lon_max_360 = AB_LON_MAX % 360

    # CASE A: 1-D lat/lon dims exist -> fast .sel slicing
    if ("latitude" in u.dims) and ("longitude" in u.dims):
        sub_u = u.sel(latitude=slice(AB_LAT_MAX, AB_LAT_MIN),
                      longitude=slice(lon_min_360, lon_max_360))
        sub_v = v.sel(latitude=slice(AB_LAT_MAX, AB_LAT_MIN),
                      longitude=slice(lon_min_360, lon_max_360))
        return sub_u, sub_v

    # CASE B: lat/lon are 2-D coords on (y,x) -> mask + drop
    latc = u.coords.get("latitude", None)
    lonc = u.coords.get("longitude", None)
    if latc is None or lonc is None:
        raise ValueError(f"Missing latitude/longitude coords. u coords: {list(u.coords)}")

    lonc_360 = (lonc % 360)

    mask = (
        (latc >= AB_LAT_MIN) & (latc <= AB_LAT_MAX) &
        (lonc_360 >= lon_min_360) & (lonc_360 <= lon_max_360)
    )

    sub_u = u.where(mask, drop=True)
    sub_v = v.where(mask, drop=True)
    return sub_u, sub_v


def _iso_z(dt64) -> str:
    """Convert numpy datetime64 to ISO8601 with Z."""
    # dt64 may be numpy.datetime64 or python datetime
    if isinstance(dt64, np.datetime64):
        # Convert to python datetime in UTC-ish representation
        ts = dt64.astype("datetime64[ms]").astype(dt.datetime)
    elif isinstance(dt64, dt.datetime):
        ts = dt64
    else:
        # fallback
        ts = dt.datetime.fromisoformat(str(dt64).replace("Z", ""))
    # Ensure it *looks* like UTC; Earth just needs a parseable time string.
    return ts.replace(tzinfo=None).isoformat(timespec="milliseconds") + "Z"


def earth_grid_from_subset(u_da: xr.DataArray, v_da: xr.DataArray, ref_time_iso: str, forecast_hour: int):
    """
    Build two Earth-compatible JSON objects (u_json, v_json) with:
      { "header": {...}, "data": [...] }
    """

    if u_da.shape != v_da.shape:
        raise ValueError(f"u and v shapes differ: {u_da.shape} vs {v_da.shape}")

    # Determine grid and orientation.
    # Earth expects:
    #   lo1 = westernmost lon
    #   la1 = northernmost lat
    #   dx  = lon spacing (degrees, positive)
    #   dy  = lat spacing, positive in header, but Earth uses header.dy and assumes lat decreases (dy positive OK if la1 is north and j computed with (φ0 - φ)/Δφ)
    #
    # In Beccario's code, it treats Δφ = header.dy, and uses j = (φ0 - φ) / Δφ, so Δφ should be positive.
    # So set dy = (lat_max - lat_min) / (ny-1) > 0, and la1 = lat_max.

    # Case A: regular lat/lon dimension coords
    if ("latitude" in u_da.dims) and ("longitude" in u_da.dims):
        lats = u_da["latitude"].values
        lons = u_da["longitude"].values

        # Convert lons to -180..180 for Earth
        lons = np.where(lons > 180, lons - 360, lons)

        # Ensure lats are north->south and lons west->east
        # After your .sel(latitude=slice(max,min)), lats should already be descending.
        # We'll enforce it just in case.
        if lats[0] < lats[-1]:
            u_da = u_da.isel(latitude=slice(None, None, -1))
            v_da = v_da.isel(latitude=slice(None, None, -1))
            lats = u_da["latitude"].values

        if lons[0] > lons[-1]:
            u_da = u_da.isel(longitude=slice(None, None, -1))
            v_da = v_da.isel(longitude=slice(None, None, -1))
            lons = np.where(u_da["longitude"].values > 180, u_da["longitude"].values - 360, u_da["longitude"].values)

        ny = int(u_da.sizes["latitude"])
        nx = int(u_da.sizes["longitude"])

        lat_max = float(np.nanmax(lats))
        lon_min = float(np.nanmin(lons))

        dx = float((np.nanmax(lons) - np.nanmin(lons)) / (nx - 1)) if nx > 1 else 0.0
        dy = float((lat_max - float(np.nanmin(lats))) / (ny - 1)) if ny > 1 else 0.0

        u_vals = u_da.values.astype("float32")
        v_vals = v_da.values.astype("float32")

        # Flatten row-major: [j][i] where j=0 is north row, i increases east
        u_flat = u_vals.reshape(ny * nx)
        v_flat = v_vals.reshape(ny * nx)

    # Case B: curvilinear coords on (y,x) but trimmed to a rectangle-ish subset
    elif ("y" in u_da.dims) and ("x" in u_da.dims) and ("latitude" in u_da.coords) and ("longitude" in u_da.coords):
        lat2 = u_da["latitude"].values
        lon2 = u_da["longitude"].values
        lon2 = np.where(lon2 > 180, lon2 - 360, lon2)

        ny, nx = u_da.shape

        # Approximate a regular lat/lon grid for Earth:
        # Take first column for lat progression and first row for lon progression.
        # This is valid if your subset is on a regular RLatLon grid (HRDPS is).
        lat_col = lat2[:, 0]
        lon_row = lon2[0, :]

        # Ensure north->south (lat decreasing)
        if lat_col[0] < lat_col[-1]:
            u_da = u_da.isel(y=slice(None, None, -1))
            v_da = v_da.isel(y=slice(None, None, -1))
            lat2 = u_da["latitude"].values
            lon2 = np.where(u_da["longitude"].values > 180, u_da["longitude"].values - 360, u_da["longitude"].values)
            lat_col = lat2[:, 0]
            lon_row = lon2[0, :]

        # Ensure west->east (lon increasing)
        if lon_row[0] > lon_row[-1]:
            u_da = u_da.isel(x=slice(None, None, -1))
            v_da = v_da.isel(x=slice(None, None, -1))
            lat2 = u_da["latitude"].values
            lon2 = np.where(u_da["longitude"].values > 180, u_da["longitude"].values - 360, u_da["longitude"].values)
            lat_col = lat2[:, 0]
            lon_row = lon2[0, :]

        lat_max = float(np.nanmax(lat2))
        lon_min = float(np.nanmin(lon2))

        dx = float((float(np.nanmax(lon_row)) - float(np.nanmin(lon_row))) / (nx - 1)) if nx > 1 else 0.0
        dy = float((lat_max - float(np.nanmin(lat_col))) / (ny - 1)) if ny > 1 else 0.0

        u_vals = u_da.values.astype("float32")
        v_vals = v_da.values.astype("float32")

        u_flat = u_vals.reshape(ny * nx)
        v_flat = v_vals.reshape(ny * nx)

    else:
        raise ValueError(f"Unsupported dims/coords for Earth export. dims={u_da.dims}, coords={list(u_da.coords)}")

    # Replace NaN/Inf with null
    u_flat = u_flat.astype(object)
    v_flat = v_flat.astype(object)

    u_bad = ~np.isfinite(np.asarray(u_flat, dtype=float))
    v_bad = ~np.isfinite(np.asarray(v_flat, dtype=float))
    u_flat[u_bad] = None
    v_flat[v_bad] = None

    header = {
        "lo1": lon_min,
        "la1": lat_max,
        "dx": dx,
        "dy": dy,
        "nx": nx,
        "ny": ny,
        "refTime": ref_time_iso,
        "forecastTime": int(forecast_hour),
        "centerName": "ECCC HRDPS"
    }

    u_json = {"header": header, "data": u_flat.tolist()}
    v_json = {"header": header, "data": v_flat.tolist()}
    return u_json, v_json


def pick_run_cycle(now: dt.datetime | None = None) -> dt.datetime:
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
    date_str = run_time.strftime("%Y%m%d")
    cycle_hour = run_time.hour
    cycle_dir = f"{cycle_hour:02d}"
    cycle_tag = f"{cycle_dir}Z"
    lead_str = f"{lead_hour:03d}"

    filename = (
        f"{date_str}T{cycle_tag}_MSC_HRDPS_{var}_AGL-10m_RLatLon0.0225_PT{lead_str}H.grib2"
    )
    return f"{HRDPS_BASE}/{cycle_dir}/{lead_str}/{filename}"


def main():
    run_time = pick_run_cycle()
    for _ in range(2):  # current cycle, then previous cycle
        test_url = build_hrdps_url(run_time, "UGRD", 0)
        r = requests.head(test_url, timeout=30)
        if r.status_code == 200:
            break
        run_time = run_time - dt.timedelta(hours=6)

    print("Using HRDPS run cycle:", run_time.isoformat())

    lead_hours = [0, 1, 2, 3]

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    # Reference time should be the model cycle time in ISO with Z
    ref_time_iso = run_time.replace(tzinfo=None).isoformat(timespec="milliseconds") + "Z"

    for lead_hour in lead_hours:
        print(f"Processing lead hour {lead_hour}h")

        ugrd_url = build_hrdps_url(run_time, "UGRD", lead_hour)
        vgrd_url = build_hrdps_url(run_time, "VGRD", lead_hour)

        u_path = v_path = None
        try:
            u_path = download_to_temp(ugrd_url)
            v_path = download_to_temp(vgrd_url)

            u10, v10 = open_hrdps_10m_uv(u_path, v_path)

            u_json, v_json = earth_grid_from_subset(u10, v10, ref_time_iso=ref_time_iso, forecast_hour=lead_hour)

            u_out = os.path.join(out_dir, f"AB_u_{lead_hour:03d}.json")
            v_out = os.path.join(out_dir, f"AB_v_{lead_hour:03d}.json")

            with open(u_out, "w") as f:
                json.dump(u_json, f, allow_nan=False)
            with open(v_out, "w") as f:
                json.dump(v_json, f, allow_nan=False)

            print(f"Wrote {u_out}")
            print(f"Wrote {v_out}")


        finally:
            if u_path and os.path.exists(u_path):
                os.remove(u_path)
            if v_path and os.path.exists(v_path):
                os.remove(v_path)


if __name__ == "__main__":
    main()
