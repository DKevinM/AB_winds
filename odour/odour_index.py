# odour_index.py

import datetime as dt
import json
from pathlib import Path

from backtraj_core import (
    MetStore,
    run_back_trajectories,
    centerlines_to_geojson,
    cloud_to_geojson
)

# ================= CONFIG =================
MET_FOLDER = "met_data"
OUTPUT_FOLDER = "odour_output"

DEFAULT_BACK_HOURS = 4.0
DEFAULT_DT = 60
DEFAULT_PARTICLES = 200
DEFAULT_HEIGHTS = [10.0, 40.0, 80.0]
# ========================================


def run_odour_index(
    lat: float,
    lon: float,
    time_utc: dt.datetime,
    back_hours: float = DEFAULT_BACK_HOURS
):
    """
    Core odour investigation entry point.
    """

    print("Loading meteorology...")
    met = MetStore(MET_FOLDER)

    print("Running back trajectories...")
    centerlines, cloud = run_back_trajectories(
        met=met,
        start_lat=lat,
        start_lon=lon,
        start_time_utc=time_utc,
        hours=back_hours,
        dt_s=DEFAULT_DT,
        n_particles=DEFAULT_PARTICLES,
        start_heights_m=DEFAULT_HEIGHTS
    )

    print("Converting to GeoJSON...")
    center_geo = centerlines_to_geojson(centerlines)
    cloud_geo = cloud_to_geojson(cloud, every_n=3)

    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

    timestamp = time_utc.strftime("%Y%m%d_%H%M")

    center_path = Path(OUTPUT_FOLDER) / f"centerline_{timestamp}.geojson"
    cloud_path  = Path(OUTPUT_FOLDER) / f"cloud_{timestamp}.geojson"

    with open(center_path, "w") as f:
        json.dump(center_geo, f)

    with open(cloud_path, "w") as f:
        json.dump(cloud_geo, f)

    print("Saved outputs:")
    print(center_path)
    print(cloud_path)

    return center_path, cloud_path


# ========== CLI Test Runner ==========
if __name__ == "__main__":
    import os

    lat = float(os.environ["LAT"])
    lon = float(os.environ["LON"])
    time_utc = dt.datetime.fromisoformat(os.environ["TIME_UTC"])
    hours = float(os.environ["HOURS"])

    run_odour_index(lat, lon, time_utc, hours)

