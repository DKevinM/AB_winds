# src/dsai/batch_trajectories.py
#
# Runs a HYSPLIT trajectory once per day across a date range, for the
# multi-year PSCF/CWT climatology (as opposed to run_hysplit.py's
# single-event, multi-height ensemble for exceedance triage). Single
# height only (matches the literature's single-height convention,
# adjusted to 100m AGL - see project_dsai_hysplit memory for why: the
# noisy zone HYSPLIT struggles with is roughly the lowest 10-50m, not
# 100m, and 100m/500m agreed on the real July 18 case).
#
# Disk-safe: GDAS archive files are downloaded per-week, every
# trajectory needing that week is run, then the file is deleted before
# moving on - 8.5 years of archive data all at once would be ~265GB,
# too much to hold against this server's free disk.

import argparse
import datetime as dt
import os
import shutil
import subprocess
from collections import defaultdict

from stations import STATIONS
from gdas_fetch import (
    files_needed_for_event, ensure_downloaded, archive_available, MET_DIR,
)

HYSPLIT_EXEC = "/opt/airquality/hysplit/hysplit.v5.4.2_UbuntuOS20.04.6LTS/exec/hyts_std"
ASCDATA_SRC = "/opt/airquality/hysplit/hysplit.v5.4.2_UbuntuOS20.04.6LTS/bdyfiles/ASCDATA.CFG"
BATCH_RUNS_DIR = "/opt/airquality/dsai_data/hysplit_batch"
HEIGHT_M = 100
DURATION_HOURS = 72
TRAJ_HOUR_UTC = 19  # ~noon Mountain time, matching Kindzierski's own convention


def daterange(start_date, end_date, step_days=1):
    d = start_date
    while d <= end_date:
        yield d
        d += dt.timedelta(days=step_days)


def build_control(work_dir, station, event_dt, met_files):
    lat, lon = STATIONS[station]
    control_path = os.path.join(work_dir, "CONTROL")
    tdump_name = "tdump"

    lines = [
        f"{event_dt.year % 100:02d} {event_dt.month:02d} {event_dt.day:02d} {event_dt.hour:02d}",
        "1",
        f"{lat:.4f} {lon:.4f} {HEIGHT_M}.0",
        str(-DURATION_HOURS),
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


def run_one_day(station, event_dt, out_dir):
    met_files = files_needed_for_event(event_dt, duration_hours=DURATION_HOURS)
    for f in met_files:
        ensure_downloaded(f)

    work_dir = os.path.join(out_dir, event_dt.strftime("%Y%m%d"))
    os.makedirs(work_dir, exist_ok=True)
    shutil.copy(ASCDATA_SRC, work_dir)

    build_control(work_dir, station, event_dt, met_files)
    result = subprocess.run([HYSPLIT_EXEC], cwd=work_dir, capture_output=True, text=True, timeout=180)

    tdump_path = os.path.join(work_dir, "tdump")
    return tdump_path if os.path.exists(tdump_path) else None, met_files


def run_batch(station, dates, out_dir, cleanup=True):
    if not archive_available(dates[-1], now=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)):
        raise RuntimeError("Latest requested date is too recent for the permanent GDAS archive - "
                            "batch climatology work should only use archived (not near-real-time) data.")

    os.makedirs(out_dir, exist_ok=True)

    # group dates by which met files they need, so each file gets
    # downloaded once, used for everything that needs it, then dropped
    files_to_dates = defaultdict(list)
    for d in dates:
        event_dt = dt.datetime.combine(d, dt.time(TRAJ_HOUR_UTC, 0))
        for f in files_needed_for_event(event_dt, duration_hours=DURATION_HOURS):
            files_to_dates[f].append(event_dt)

    all_needed_files = set(files_to_dates.keys())
    results = {}

    for d in dates:
        # resumable: a prior run may have already produced this day's
        # tdump (this is exactly how the batch got restarted after the
        # first overnight run crashed partway through)
        existing = os.path.join(out_dir, d.strftime("%Y%m%d"), "tdump")
        if os.path.exists(existing) and os.path.getsize(existing) > 0:
            results[d] = existing
            print(f"{d}  SKIP (already done)")
            continue

        event_dt = dt.datetime.combine(d, dt.time(TRAJ_HOUR_UTC, 0))
        try:
            tdump_path, used_files = run_one_day(station, event_dt, out_dir)
        except Exception as ex:
            # one bad day (e.g. a transient FTP failure that exhausted
            # retries) must not take down a multi-thousand-day batch -
            # log it and keep going, don't crash
            print(f"{d}  FAILED  ({type(ex).__name__}: {ex})")
            results[d] = None
            continue

        results[d] = tdump_path
        status = "OK" if tdump_path else "FAILED"
        print(f"{d}  {status}")

    if cleanup:
        for f in all_needed_files:
            local_path = os.path.join(MET_DIR, f)
            if os.path.exists(local_path):
                os.remove(local_path)
                print(f"Cleaned up {f}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("station")
    ap.add_argument("--sample", action="store_true",
                     help="Run ~17 dates spread evenly across the window instead of every day")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    start = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end = dt.datetime.strptime(args.end, "%Y-%m-%d").date()

    if args.sample:
        total_days = (end - start).days
        step = total_days // 16
        dates = list(daterange(start, end, step_days=step))
        out_dir = args.out_dir or os.path.join(BATCH_RUNS_DIR, "sample_check")
    else:
        dates = list(daterange(start, end))
        out_dir = args.out_dir or os.path.join(BATCH_RUNS_DIR, "full")

    print(f"Running {len(dates)} trajectories for {args.station}, {args.start} to {args.end}")
    print(f"Height: {HEIGHT_M}m AGL, duration: {DURATION_HOURS}h backward, output: {out_dir}")
    print()

    results = run_batch(args.station, dates, out_dir)

    ok = sum(1 for v in results.values() if v)
    print()
    print(f"Done: {ok}/{len(results)} succeeded")
