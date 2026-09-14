# src/dsai/rerun_deep_dive.py
#
# The escalation half of Kevin's 2026-09-14 call: default trigger runs
# are 6h (fast - see run_hysplit.py's DEFAULT_DURATION_HOURS docstring
# for the real tradeoff that bet makes). When a 6h answer looks
# ambiguous or implausible, this reruns both models - HYSPLIT and
# HRDPS - at a longer duration for that one specific event, and
# compares them, in a single command instead of three manual ones.
#
# Genuinely a separate run, not an extension of the 6h one: HYSPLIT's
# and HRDPS's own run_id/work_dir naming already includes duration_hours
# (see run_hysplit.py's run_id, run_hrdps.py's run_id), so this doesn't
# clobber the original 6h output - both are kept, side by side, if
# there's ever a reason to compare "what changed with more lookback."

import datetime as dt
import sys

from run_hysplit import run_ensemble
from run_hrdps import run_hrdps
from compare_trajectories import compare, combined_geojson, print_report

DEFAULT_DEEP_DURATION_HOURS = 24  # the previously-validated default, before Kevin's 6h speed call


def rerun_deep_dive(station, event_dt, duration_hours=DEFAULT_DEEP_DURATION_HOURS):
    print(f"Deep-dive rerun: {station} @ {event_dt.isoformat()}, {duration_hours}h back")

    print("\n--- HYSPLIT (GFS) ---")
    hysplit_results, hysplit_work_dir = run_ensemble(station, event_dt, duration_hours=duration_hours)
    if not any(v is not None for v in hysplit_results.values()):
        print("HYSPLIT produced no usable trajectory at this duration either - "
              "likely a real met-data gap (see trajectory_retry_queue.py), not something a longer run fixes.")
        return

    print("\n--- HRDPS (particle, terrain-steered) ---")
    centerlines_path = run_hrdps(station, event_dt, duration_hours)
    if not centerlines_path:
        print("HRDPS produced no usable output at this duration (outside its "
              "~30-day wind-data retention, or a real run failure above) - "
              "HYSPLIT result above still stands on its own.")
        return
    print(f"OK -> {centerlines_path}")

    print("\n--- Comparison ---")
    rows = compare(hysplit_work_dir, centerlines_path)
    print_report(rows, station)
    geo = combined_geojson(hysplit_work_dir, centerlines_path)
    import json
    import os
    out_path = os.path.join(hysplit_work_dir, "dual_model_comparison.geojson")
    with open(out_path, "w") as fh:
        json.dump(geo, fh)
    print(f"\nCombined overlay: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: rerun_deep_dive.py '<Station Name>' <YYYY-MM-DDTHH:MM> [duration_hours=24]")
        sys.exit(1)

    station = sys.argv[1]
    event_dt = dt.datetime.strptime(sys.argv[2], "%Y-%m-%dT%H:%M")
    duration = int(sys.argv[3]) if len(sys.argv) == 4 else DEFAULT_DEEP_DURATION_HOURS

    rerun_deep_dive(station, event_dt, duration_hours=duration)
