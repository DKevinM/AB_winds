# build_ab_wind_json.py
import json
import datetime as dt
import xarray as xr

# Rough Alberta bounding box
AB_LAT_MIN, AB_LAT_MAX = 48.5, 60.5
AB_LON_MIN, AB_LON_MAX = -120.0, -108.0  # adjust if you want more margin

def open_gfs_grib(path):
    """
    Open a GFS GRIB2 file and extract 10 m u/v wind components as xarray DataArrays.
    """
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "typeOfLevel": "heightAboveGround",
                "level": 10
            }
        },
    )
    # Variable names may be 'u10', 'v10', or '10u', '10v' depending on source
    for u_name in ["u10", "10u", "UGRD"]:
        if u_name in ds:
            u = ds[u_name]
            break
    else:
        raise ValueError("No U-wind found")

    for v_name in ["v10", "10v", "VGRD"]:
        if v_name in ds:
            v = ds[v_name]
            break
    else:
        raise ValueError("No V-wind found")

    # Subset to Alberta bounding box
    sub = ds.sel(
        latitude=slice(AB_LAT_MAX, AB_LAT_MIN),   # note: descending lat
        longitude=slice(AB_LON_MIN % 360, AB_LON_MAX % 360)
    )

    u_sub = sub[u.name]
    v_sub = sub[v.name]

    # Use the first time slice (you can extend to multiple later)
    if "time" in u_sub.dims:
        u_sub = u_sub.isel(time=0)
        v_sub = v_sub.isel(time=0)

    return u_sub, v_sub


def to_earth_like_json(u, v):
    """
    Convert u/v DataArrays into a compact JSON structure the JS side can read.
    """
    lats = u["latitude"].values
    lons = u["longitude"].values

    # Ensure consistent shapes
    assert u.shape == v.shape

    # We’ll flatten row-major (lat index fastest or lon fastest – just keep consistent)
    # Here: lat index first, lon second
    u_vals = u.values.astype("float32").tolist()
    v_vals = v.values.astype("float32").tolist()

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

    return {
        "meta": meta,
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "u": u_vals,
        "v": v_vals
    }


def main():
    # --- 1) Decide which GFS file to use ---
    # For now, just point to a local GRIB2 you’ve downloaded.
    grib_path = "gfs_10m_wind_example.grib2"

    u10, v10 = open_gfs_grib(grib_path)
    js = to_earth_like_json(u10, v10)

    # Name output as "latest" for now – GitHub Actions can overwrite it
    out_path = "data/AB_wind_latest.json"
    with open(out_path, "w") as f:
        json.dump(js, f)

    print("Wrote", out_path)


if __name__ == "__main__":
    main()
