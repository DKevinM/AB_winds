# src/dsai/run_hysplit.py
#
# Given a station name and event timestamp, runs a 3-height HYSPLIT
# backward-trajectory ensemble (100/500/1000m AGL) - the same setup
# hand-built and validated against the real 2026-07-18 Edmonton East H2S
# event. Auto-fetches whatever met data is needed - the permanent GDAS
# archive if it's old enough to be posted, or near-real-time gfsa cycles
# for anything too recent for that (see gdas_fetch.py).
#
# Default duration is 6h - Kevin's call 2026-09-14: a fast first look
# (fewer met files, ~4x quicker HYSPLIT run, and the paired HRDPS run
# drops from ~7min to ~2min) that's usually enough to answer "where did
# this come from," with a longer run only when the 6h answer looks
# ambiguous or implausible. See rerun_deep_dive.py for that escalation
# step - reruns both models (and the comparison) at a longer duration
# for one specific already-triggered event.
#
# Real methodology risk worth remembering if 6h answers keep looking
# wrong: the one case validated against ground truth (2026-07-18
# Edmonton East H2S) needed a 27h window to resolve the actual delivery
# mechanism - the parcel sat at ground level for ~30h before a gust
# front lofted and delivered it. A 6h default would have missed that
# story entirely; it was only found by running the full 24h/72h
# ensemble. 6h is a real bet that most triggered events are nearer-field
# than that one was - reasonable given both models already tend to
# agree closely in the first few hours and only diverge later (see
# compare_trajectories.py's real output on 2026-09-13/14), but not a
# guarantee.

import os
import subprocess
import datetime as dt

from stations import STATIONS
from gdas_fetch import ensure_met_files_for_event, MET_DIR

# tdump header is deterministic: 1 (grid count) + 1 per met file (GFSG
# line) + 1 (BACKWARD/FORWARD line) + 1 (start point line) + 1 (output
# variable count line) - a file with nothing beyond that many lines
# has zero actual trajectory points, which HYSPLIT can produce (exit
# code 0, file created) even when the run failed internally, e.g.
# "*ERROR* metpos: start point not within (x,y,t) any data file" when
# the requested start time falls in a real gap in the fetched met data
# (near-real-time gfsa cycles routinely lag several hours behind
# wall-clock - confirmed live 2026-09-13, the freshest published data
# was ~9h stale). Originally only checked os.path.exists(), which this
# empty-but-present file always satisfies - every trigger silently
# logged "OK" with no usable trajectory data behind it. Fixed 2026-09-13.
HEADER_LINES_PER_HEIGHT_OVERHEAD = 4  # grid-count + BACKWARD line + start-point line + var-count line


def _tdump_has_trajectory(tdump_path, n_met_files):
    if not os.path.exists(tdump_path):
        return False
    with open(tdump_path) as fh:
        line_count = sum(1 for _ in fh)
    return line_count > (n_met_files + HEADER_LINES_PER_HEIGHT_OVERHEAD)


def _extract_hysplit_error(result):
    """Pull the real diagnostic line out of hyts_std's output, if any -
    e.g. "*ERROR* metpos: start point not within (x,y,t) any data
    file" - so a failure log says why, not just that it failed."""
    combined = (result.stdout or "") + (result.stderr or "")
    for line in combined.splitlines():
        if "*ERROR*" in line or "ERROR" in line:
            return line.strip()
    return None

HYSPLIT_EXEC = "/opt/airquality/hysplit/hysplit.v5.4.2_UbuntuOS20.04.6LTS/exec/hyts_std"
ASCDATA_SRC = "/opt/airquality/hysplit/hysplit.v5.4.2_UbuntuOS20.04.6LTS/bdyfiles/ASCDATA.CFG"
RUNS_DIR = "/opt/airquality/dsai_data/hysplit_runs"
HEIGHTS_M = [100, 500, 1000]
DEFAULT_DURATION_HOURS = 6


def resolve_station(station_or_coord):
    """station_or_coord: either a STATIONS key (str, existing behavior -
    every current caller passes one of these) or a (lat, lon, label)
    tuple for an ad-hoc location not in STATIONS (e.g. a Whitecap
    incident site in Saskatchewan - added 2026-09-16 for the Whitecap
    incident-trajectory watcher). Returns (lat, lon, label)."""
    if isinstance(station_or_coord, tuple):
        lat, lon, label = station_or_coord
        return lat, lon, label
    if station_or_coord not in STATIONS:
        raise ValueError(f"Unknown station: {station_or_coord}")
    lat, lon = STATIONS[station_or_coord]
    return lat, lon, station_or_coord


def build_control(work_dir, lat, lon, event_dt, height_m, duration_hours, met_files, direction="backward"):
    """met_files: list of (filename, subdir) tuples - subdir is "" for the
    permanent archive (lives directly in MET_DIR) or a YYYYMMDD subdir
    for near-real-time gfsa cycles.

    direction: "backward" (default - HYSPLIT traces where air arriving
    at (lat, lon) at event_dt came from) or "forward" (added 2026-09-17 -
    traces where a release starting at (lat, lon) at event_dt goes to;
    HYSPLIT's own CONTROL format already supports this - it's just the
    sign of the duration line - so this isn't new modeling, just
    exposing a mode the binary already has). Use forward whenever
    (lat, lon) IS the known source (e.g. a spill/leak site) rather than
    a receptor looking for an upwind source."""
    control_path = os.path.join(work_dir, f"CONTROL_{height_m}m")
    tdump_name = f"tdump_{height_m}m"

    signed_duration = abs(duration_hours) if direction == "forward" else -abs(duration_hours)
    lines = [
        f"{event_dt.year % 100:02d} {event_dt.month:02d} {event_dt.day:02d} {event_dt.hour:02d}",
        "1",
        f"{lat:.4f} {lon:.4f} {height_m}.0",
        str(signed_duration),  # negative = backward, positive = forward
        "0",
        "10000.0",
        str(len(met_files)),
    ]
    for fname, subdir in met_files:
        met_path = os.path.join(MET_DIR, subdir) if subdir else MET_DIR
        lines.append(f"{met_path}/")
        lines.append(fname)
    lines.append(f"{work_dir}/")
    lines.append(tdump_name)

    with open(control_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    return control_path, tdump_name


def run_ensemble(station, event_dt, duration_hours=DEFAULT_DURATION_HOURS, direction="backward", heights_m=None):
    """
    station: key into STATIONS, OR a (lat, lon, label) tuple for an
    ad-hoc location not in STATIONS.
    event_dt: naive datetime in UTC of the flagged reading
    duration_hours: how far to run (24 default, 72 for deep-dive)
    direction: "backward" (default - where did the air here come from;
    for a receptor investigating an upwind source) or "forward" (added
    2026-09-17 - where does a release from here go; for a known source
    like a spill/leak site - see build_control's docstring)
    heights_m: override for the module's HEIGHTS_M = [100, 500, 1000]
    default (added 2026-09-17). That default was chosen for the backward
    receptor use case - bracketing plausible altitudes a plume might be
    traveling at when it reaches a station, since it could have been
    lofted well above ground (see the real July 18 gust-front case in
    this module's history). None of those heights make sense as a
    *forward* run's starting point for a ground-level source like a
    Whitecap pipeline leak or wellsite spill - the release starts at the
    ground, not already 500-1000m up. Kevin's call 2026-09-17: forward
    Whitecap runs use [10, 100] instead (both HYSPLIT and HRDPS, for
    consistency between the two models).
    Returns (results dict of {height_m: tdump_file_path}, work_dir)
    """
    lat, lon, label = resolve_station(station)
    heights_m = heights_m or HEIGHTS_M

    met_files = ensure_met_files_for_event(event_dt, duration_hours=duration_hours, direction=direction)

    # Suffix only for forward (the new mode) so every existing backward
    # caller's work_dir naming is byte-identical to before 2026-09-17 -
    # not just the trajectory output.
    run_id = f"{label.replace(' ', '_')}_{event_dt.strftime('%Y%m%dT%H%M')}_{duration_hours}h"
    if direction == "forward":
        run_id += "_forward"
    work_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(work_dir, exist_ok=True)
    subprocess.run(["cp", ASCDATA_SRC, work_dir], check=True)

    results = {}
    for h in heights_m:
        control_path, tdump_name = build_control(work_dir, lat, lon, event_dt, h, duration_hours, met_files, direction=direction)
        # hyts_std reads "CONTROL" from cwd by default - point it at ours via symlink
        cwd_control = os.path.join(work_dir, "CONTROL")
        if os.path.exists(cwd_control) or os.path.islink(cwd_control):
            os.remove(cwd_control)
        os.symlink(control_path, cwd_control)

        print(f"Running HYSPLIT for {station} @ {h}m AGL, {duration_hours}h {direction} ...")
        result = subprocess.run(
            [HYSPLIT_EXEC], cwd=work_dir, capture_output=True, text=True, timeout=300
        )
        tdump_path = os.path.join(work_dir, tdump_name)
        if _tdump_has_trajectory(tdump_path, len(met_files)):
            print(f"  OK -> {tdump_path}")
            results[h] = tdump_path
        else:
            error_line = _extract_hysplit_error(result)
            if error_line:
                print(f"  FAILED - {error_line}")
            else:
                print(f"  FAILED - no trajectory points in output. stdout tail:\n{result.stdout[-500:]}")
            results[h] = None

    return results, work_dir


if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (3, 4):
        print("Usage: run_hysplit.py '<Station Name>' <YYYY-MM-DDTHH:MM> [duration_hours]")
        sys.exit(1)

    station = sys.argv[1]
    event_dt = dt.datetime.strptime(sys.argv[2], "%Y-%m-%dT%H:%M")
    duration = int(sys.argv[3]) if len(sys.argv) == 4 else DEFAULT_DURATION_HOURS

    results, work_dir = run_ensemble(station, event_dt, duration_hours=duration)
    print()
    print(f"Run directory: {work_dir}")
    for h, path in results.items():
        print(f"  {h}m: {path}")
