from datetime import datetime, timezone
from odour.config import TrajectoryConfig
from odour.wind_field import WindCache, WindField
from odour.trajectory import back_trajectory_ensemble

# TODO: implement this mapping based on how you cache hours
def slot_map(dt_hour):
    # simplest: always use "AB_wind_latest.json" for prototype
    # (later: map dt_hour -> AB_wind_000/001/002/003)
    return "AB_wind_latest.json"

def main():
    lon0, lat0 = -113.4909, 53.5444  # example
    t0 = datetime.now(timezone.utc)

    cache = WindCache("data/winds")
    wf = WindField(cache, slot_map)

    cfg = TrajectoryConfig(dt_seconds=120, lookback_hours=5, n_particles=200)

    paths = back_trajectory_ensemble(lon0, lat0, t0, wf, cfg)
    print("paths:", len(paths), "steps:", len(paths[0].lon))

if __name__ == "__main__":
    main()
