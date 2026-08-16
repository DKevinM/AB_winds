# src/dsai/run_hysplit.py
#
# Given a station name and event timestamp, runs a 3-height HYSPLIT
# backward-trajectory ensemble (100/500/1000m AGL, 72h) - the same setup
# hand-built and validated against the real 2026-07-18 Edmonton East H2S
# event. Auto-fetches whatever GDAS files are needed first.

import os
import subprocess
import datetime as dt

from stations import STATIONS
from gdas_fetch import ensure_met_files_for_event, MET_DIR

HYSPLIT_EXEC = "/opt/airquality/hysplit/hysplit.v5.4.2_UbuntuOS20.04.6LTS/exec/hyts_std"
ASCDATA_SRC = "/opt/airquality/hysplit/hysplit.v5.4.2_UbuntuOS20.04.6LTS/bdyfiles/ASCDATA.CFG"
RUNS_DIR = "/opt/airquality/dsai_data/hysplit_runs"
HEIGHTS_M = [100, 500, 1000]
DURATION_HOURS = -72  # negative = backward


def build_control(work_dir, station, event_dt, height_m, met_files):
    lat, lon = STATIONS[station]
    control_path = os.path.join(work_dir, f"CONTROL_{height_m}m")
    tdump_name = f"tdump_{height_m}m"

    lines = [
        f"{event_dt.year % 100:02d} {event_dt.month:02d} {event_dt.day:02d} {event_dt.hour:02d}",
        "1",
        f"{lat:.4f} {lon:.4f} {height_m}.0",
        str(DURATION_HOURS),
        "0",
        "10000.0",
        str(len(met_files)),
    ]
    for f in met_files:
        lines.append(f"{MET_DIR}/")
        lines.append(f)
    lines.append(f"{work_dir}/")
    lines.append(tdump_name)

    with open(control_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    return control_path, tdump_name


def run_ensemble(station, event_dt):
    """
    station: key into STATIONS
    event_dt: naive datetime in UTC of the flagged reading
    Returns dict of {height_m: tdump_file_path}
    """
    if station not in STATIONS:
        raise ValueError(f"Unknown station: {station}")

    met_files = ensure_met_files_for_event(event_dt)

    run_id = f"{station.replace(' ', '_')}_{event_dt.strftime('%Y%m%dT%H%M')}"
    work_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(work_dir, exist_ok=True)
    subprocess.run(["cp", ASCDATA_SRC, work_dir], check=True)

    results = {}
    for h in HEIGHTS_M:
        control_path, tdump_name = build_control(work_dir, station, event_dt, h, met_files)
        # hyts_std reads "CONTROL" from cwd by default - point it at ours via symlink
        cwd_control = os.path.join(work_dir, "CONTROL")
        if os.path.exists(cwd_control) or os.path.islink(cwd_control):
            os.remove(cwd_control)
        os.symlink(control_path, cwd_control)

        print(f"Running HYSPLIT for {station} @ {h}m AGL ...")
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

    if len(sys.argv) != 3:
        print("Usage: run_hysplit.py '<Station Name>' <YYYY-MM-DDTHH:MM>")
        sys.exit(1)

    station = sys.argv[1]
    event_dt = dt.datetime.strptime(sys.argv[2], "%Y-%m-%dT%H:%M")

    results, work_dir = run_ensemble(station, event_dt)
    print()
    print(f"Run directory: {work_dir}")
    for h, path in results.items():
        print(f"  {h}m: {path}")
