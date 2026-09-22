# src/dsai/run_click_trajectory.py
#
# Worker for the Whitecap status map's "click anywhere, run a forward
# trajectory" feature (added 2026-09-22) - called as a subprocess by
# server/main.py's job queue, same env-var-driven __main__ convention as
# odour/backtraj_core.py (LAT/LON/TIME_LOCAL/HOURS/OUTDIR), so the
# existing FastAPI job-submit/poll infrastructure could be reused
# as-is for a second endpoint rather than building a parallel one.
#
# Deliberately NOT backtraj_core.py's custom particle-cloud model -
# Kevin's call was explicit ("run it for 12hrs at 10m and 100m as
# usual"): this uses the exact same real HYSPLIT engine
# (run_hysplit.run_ensemble) already validated and in production for
# Whitecap's real incident trajectories (watch_whitecap_incidents.py),
# just triggered from an arbitrary clicked point instead of a real
# incident's coordinates. Same forward direction, same [10, 100]m
# ground-level-source heights, same tdump_to_geojson converter - a
# hypothetical "what if a release happened here" run should look and
# behave exactly like a real one does elsewhere on this page.
#
# TIME_LOCAL is Saskatchewan time here (America/Regina, fixed CST -
# never DST, unlike odour/backtraj_core.py's America/Edmonton), since
# that's what a Whitecap-page viewer types in.

import datetime as dt
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from run_hysplit import run_ensemble
from watch_whitecap_incidents import tdump_to_geojson

SASKATCHEWAN_TZ = ZoneInfo("America/Regina")
DURATION_HOURS = 12
HEIGHTS_M = [10, 100]


if __name__ == "__main__":
    lat = float(os.environ["LAT"])
    lon = float(os.environ["LON"])
    time_local = os.environ["TIME_LOCAL"]

    naive_local = dt.datetime.fromisoformat(time_local)
    local_dt = naive_local.replace(tzinfo=SASKATCHEWAN_TZ)
    event_dt = local_dt.astimezone(dt.timezone.utc).replace(tzinfo=None)

    label = "click"
    outdir = Path(os.environ.get("OUTDIR", "click_trajectory_data"))
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Running HYSPLIT for a clicked point @ {lat:.4f},{lon:.4f}, "
          f"{time_local} Saskatchewan time ({DURATION_HOURS}h forward) ...")
    try:
        results, _work_dir = run_ensemble(
            (lat, lon, label), event_dt,
            duration_hours=DURATION_HOURS, direction="forward", heights_m=HEIGHTS_M,
        )
    except RuntimeError:
        # ensure_met_files_for_event raises this when near-real-time met
        # data simply doesn't cover event_dt at all (as opposed to
        # covering it but producing an empty trajectory, handled below) -
        # same real/expected gap, just caught earlier in run_ensemble's
        # own pipeline. Same clean error.json treatment either way, so
        # the FastAPI job status reports a plain message instead of a
        # raw Python traceback (found via headless-browser testing).
        with open(outdir / "error.json", "w") as f:
            json.dump({"error": "No near-real-time met data covers this start "
                                 "time yet - it's likely too recent (met data "
                                 "typically lags by up to ~9 hours) or too far "
                                 "in the past for the archive still covering it. "
                                 "Try a time from a few hours to a few days ago."}, f)
        raise SystemExit(1)

    if not any(v is not None for v in results.values()):
        # Real, expected failure mode - same as the incident watcher's
        # own handling: near-real-time met data can lag wall-clock by
        # ~9h+, so a start time in that gap genuinely has nothing to
        # run against yet. Surfaced as a clean error file, not a raw
        # exception, so the FastAPI job status can report it plainly.
        with open(outdir / "error.json", "w") as f:
            json.dump({"error": "HYSPLIT produced no usable trajectory - "
                                 "the requested start time may be too recent "
                                 "for met data to be published yet, or too far "
                                 "in the past for the near-real-time archive "
                                 "still covering it."}, f)
        raise SystemExit(1)

    geojson = tdump_to_geojson(results, label)
    with open(outdir / "trajectory.geojson", "w") as f:
        json.dump(geojson, f)
    print(f"-> {outdir / 'trajectory.geojson'}")
