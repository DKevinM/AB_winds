# src/dsai/run_hrdps.py
#
# Thin wrapper around odour/backtraj_core.py for automated use from
# DSAI (as opposed to its normal path: a human-submitted request via
# LiveMap -> trigger_request.json -> run_odour_poll.sh). Runs it as a
# subprocess with env vars, same as that poll script does, but with
# OUTDIR pointed at a per-event directory (backtraj_core.py already
# supports this - os.environ.get("OUTDIR", "odour_data")) instead of
# the fixed odour_data/ path multiple concurrent/sequential DSAI events
# would otherwise clobber.

import os
import subprocess
import datetime as dt

from stations import STATIONS

AB_WINDS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTRAJ_SCRIPT = os.path.join(AB_WINDS_ROOT, "odour", "backtraj_core.py")
RUNS_DIR = "/opt/airquality/dsai_data/hrdps_runs"
PYTHON_EXEC = "/opt/airquality/venv/bin/python3"


def run_hrdps(station, event_dt, duration_hours, direction="backward", heights_m=None):
    """
    station: key into STATIONS, OR a (lat, lon, label) tuple for an
    ad-hoc location not in STATIONS (e.g. a Whitecap incident site -
    added 2026-09-16). event_dt: naive UTC datetime.
    direction: "backward" (default - original behavior, where did the
    air here come from) or "forward" (added 2026-09-17 - where does a
    release from here go; see backtraj_core.py's run_back_trajectories
    docstring for the underlying physics). Use forward whenever
    (lat, lon) IS the known source rather than a receptor.
    heights_m: override for run_back_trajectories' own (10, 40, 80)m
    default start heights (added 2026-09-17, alongside direction, for
    the same reason - see run_hysplit.py's run_ensemble docstring).
    Omit to keep every existing caller's heights unchanged.
    Returns the path to backtraj_centerlines.geojson on success, or
    None (with the failure reason printed) on failure - HRDPS coverage
    is real but not universal (see MetStoreV2's "No wind files found"
    for anything outside its rolling ~30-day retention), so this is
    expected to occasionally come back empty, not a bug when it does.
    """
    if isinstance(station, tuple):
        lat, lon, label = station
    else:
        if station not in STATIONS:
            raise ValueError(f"Unknown station: {station}")
        lat, lon = STATIONS[station]
        label = station

    # Suffix only for forward (the new mode) so every existing backward
    # caller's outdir naming is byte-identical to before 2026-09-17 -
    # not just the trajectory output.
    run_id = f"{label.replace(' ', '_')}_{event_dt.strftime('%Y%m%dT%H%M')}_{duration_hours}h"
    if direction == "forward":
        run_id += "_forward"
    outdir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(outdir, exist_ok=True)

    env = os.environ.copy()
    env["LAT"] = str(lat)
    env["LON"] = str(lon)
    env["TIME_UTC"] = event_dt.isoformat()
    env["HOURS"] = str(duration_hours)
    env["DIRECTION"] = direction
    if heights_m is not None:
        env["HEIGHTS_M"] = ",".join(str(h) for h in heights_m)
    env["OUTDIR"] = outdir

    # Real timed 24h run measured ~7min (2026-09-13) - 600s gives
    # genuine margin rather than the 180/300s first guessed and then
    # hit mid-run.
    result = subprocess.run(
        [PYTHON_EXEC, BACKTRAJ_SCRIPT], cwd=AB_WINDS_ROOT,
        env=env, capture_output=True, text=True, timeout=600,
    )

    centerlines_path = os.path.join(outdir, "backtraj_centerlines.geojson")
    if result.returncode != 0 or not os.path.exists(centerlines_path):
        tail = (result.stdout or "")[-600:] + (result.stderr or "")[-600:]
        print(f"  HRDPS run failed for {station} @ {event_dt}: {tail.strip()[-300:]}")
        return None
    return centerlines_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: run_hrdps.py '<Station Name>' <YYYY-MM-DDTHH:MM> <duration_hours>")
        sys.exit(1)

    station = sys.argv[1]
    event_dt = dt.datetime.strptime(sys.argv[2], "%Y-%m-%dT%H:%M")
    duration = int(sys.argv[3])

    path = run_hrdps(station, event_dt, duration)
    print(f"Centerlines: {path}")
