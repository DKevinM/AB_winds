# src/dsai/run_dual_model_comparison.py
#
# Runs every 20 minutes (see run_dsai_dual_model.sh / crontab), draining
# dual_model_queue.py - extreme exceedances (climatology.is_extreme)
# where HYSPLIT already succeeded. HYSPLIT's own work is reused (its
# work_dir was stored at enqueue time); this only pays HRDPS's cost
# (~7 minutes measured, see run_hrdps.py) plus the comparison itself
# (near-instant).

import datetime as dt

from run_hrdps import run_hrdps
from compare_trajectories import compare, combined_geojson, print_report
import dual_model_queue


def main():
    entries = dual_model_queue.pop_all()
    if not entries:
        print("No dual-model comparisons queued.")
        return

    print(f"{len(entries)} dual-model comparison(s) queued.")
    for entry in entries:
        station = entry["station"]
        event_dt = dt.datetime.fromisoformat(entry["event_dt"])
        duration_hours = entry["duration_hours"]
        hysplit_work_dir = entry["hysplit_work_dir"]

        print(f"DUAL-MODEL COMPARISON: {station} / {entry['parameter']} @ {entry['cur_ts']}")
        centerlines_path = run_hrdps(station, event_dt, duration_hours)
        if not centerlines_path:
            print(f"  Skipped - HRDPS produced no usable output for {station} @ {entry['cur_ts']} "
                  f"(likely outside its ~30-day wind-data retention, or a real run failure - see above)")
            continue

        try:
            rows = compare(hysplit_work_dir, centerlines_path)
            print_report(rows, station)
            geo = combined_geojson(hysplit_work_dir, centerlines_path)
            import json
            import os
            out_path = os.path.join(hysplit_work_dir, "dual_model_comparison.geojson")
            with open(out_path, "w") as fh:
                json.dump(geo, fh)
            print(f"  Combined overlay: {out_path}")
        except Exception as ex:
            print(f"  Comparison failed: {ex}")


if __name__ == "__main__":
    main()
