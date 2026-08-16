# src/dsai/run_hysplit.py
#
# Given a station name and event timestamp, runs a 3-height HYSPLIT
# backward-trajectory ensemble (100/500/1000m AGL) - the same setup
# hand-built and validated against the real 2026-07-18 Edmonton East H2S
# event. Auto-fetches whatever met data is needed - the permanent GDAS
# archive if it's old enough to be posted, or near-real-time gfsa cycles
# for anything too recent for that (see gdas_fetch.py).
#
# Default duration is 24h: for single-incident triage (as opposed to the
# literature's 72h climatological convention), the actual delivery
# mechanism is usually resolved within a day - confirmed against the
# July 18 case itself, where the parcel's descent-to-ground-level and
# gust-front delivery played out entirely within a 27h window. Pass
# duration_hours=72 for cases where the longer look-back (air mass
# origin/composition, not just immediate transport) is worth the extra
# met data and runtime.

import os
import subprocess
import datetime as dt

from stations import STATIONS
from gdas_fetch import ensure_met_files_for_event, MET_DIR

HYSPLIT_EXEC = "/opt/airquality/hysplit/hysplit.v5.4.2_UbuntuOS20.04.6LTS/exec/hyts_std"
ASCDATA_SRC = "/opt/airquality/hysplit/hysplit.v5.4.2_UbuntuOS20.04.6LTS/bdyfiles/ASCDATA.CFG"
RUNS_DIR = "/opt/airquality/dsai_data/hysplit_runs"
HEIGHTS_M = [100, 500, 1000]
DEFAULT_DURATION_HOURS = 24


def build_control(work_dir, station, event_dt, height_m, duration_hours, met_files):
    """met_files: list of (filename, subdir) tuples - subdir is "" for the
    permanent archive (lives directly in MET_DIR) or a YYYYMMDD subdir
    for near-real-time gfsa cycles."""
    lat, lon = STATIONS[station]
    control_path = os.path.join(work_dir, f"CONTROL_{height_m}m")
    tdump_name = f"tdump_{height_m}m"

    lines = [
        f"{event_dt.year % 100:02d} {event_dt.month:02d} {event_dt.day:02d} {event_dt.hour:02d}",
        "1",
        f"{lat:.4f} {lon:.4f} {height_m}.0",
        str(-abs(duration_hours)),  # negative = backward
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


def run_ensemble(station, event_dt, duration_hours=DEFAULT_DURATION_HOURS):
    """
    station: key into STATIONS
    event_dt: naive datetime in UTC of the flagged reading
    duration_hours: how far back to run (24 default, 72 for deep-dive)
    Returns (results dict of {height_m: tdump_file_path}, work_dir)
    """
    if station not in STATIONS:
        raise ValueError(f"Unknown station: {station}")

    met_files = ensure_met_files_for_event(event_dt, duration_hours=duration_hours)

    run_id = f"{station.replace(' ', '_')}_{event_dt.strftime('%Y%m%dT%H%M')}_{duration_hours}h"
    work_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(work_dir, exist_ok=True)
    subprocess.run(["cp", ASCDATA_SRC, work_dir], check=True)

    results = {}
    for h in HEIGHTS_M:
        control_path, tdump_name = build_control(work_dir, station, event_dt, h, duration_hours, met_files)
        # hyts_std reads "CONTROL" from cwd by default - point it at ours via symlink
        cwd_control = os.path.join(work_dir, "CONTROL")
        if os.path.exists(cwd_control) or os.path.islink(cwd_control):
            os.remove(cwd_control)
        os.symlink(control_path, cwd_control)

        print(f"Running HYSPLIT for {station} @ {h}m AGL, {duration_hours}h back ...")
        result = subprocess.run(
            [HYSPLIT_EXEC], cwd=work_dir, capture_output=True, text=True, timeout=300
        )
        tdump_path = os.path.join(work_dir, tdump_name)
        if not os.path.exists(tdump_path):
            print(f"  FAILED - no tdump produced. stdout tail:\n{result.stdout[-500:]}")
            results[h] = None
        else:
            print(f"  OK -> {tdump_path}")
            results[h] = tdump_path

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
