# src/dsai/gdas_fetch.py
#
# Auto-fetches whichever GDAS weekly archive file(s) a HYSPLIT run needs,
# skipping the download if already present locally. NOAA's weekly split
# (confirmed against the live archive listing, 2026-08-16):
#   W1 = days 1-7, W2 = 8-14, W3 = 15-21, W4 = 22-28, W5 = 29-end.
# A 72-hour backward trajectory can span into the previous week if the
# event falls near a week's start, so this always fetches the event's
# own week plus the week before it.

import calendar
import os
import subprocess
import datetime as dt

MET_DIR = "/opt/airquality/hysplit/met_data"
FTP_BASE = "ftp://arlftp.arlhq.noaa.gov/pub/archives/gdas1"

MONTH_ABBR = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
}


def week_of_month(day):
    if day <= 7:
        return 1
    if day <= 14:
        return 2
    if day <= 21:
        return 3
    if day <= 28:
        return 4
    return 5


def gdas_filename(year, month, week):
    yy = year % 100
    return f"gdas1.{MONTH_ABBR[month]}{yy:02d}.w{week}"


def week_start_date(year, month, week):
    starts = {1: 1, 2: 8, 3: 15, 4: 22, 5: 29}
    return dt.date(year, month, starts[week])


def previous_week_file(year, month, week):
    """The gdas filename for the week immediately before (year, month, week)."""
    start = week_start_date(year, month, week)
    prev_day = start - dt.timedelta(days=1)
    return gdas_filename(prev_day.year, prev_day.month, week_of_month(prev_day.day))


def files_needed_for_event(event_dt):
    """
    event_dt: timezone-aware or naive datetime (UTC) of the event.
    Returns the list of gdas filenames needed to cover a 72h backward
    trajectory starting at event_dt (its own week + the prior week).
    """
    week = week_of_month(event_dt.day)
    current = gdas_filename(event_dt.year, event_dt.month, week)
    previous = previous_week_file(event_dt.year, event_dt.month, week)
    # de-dupe while preserving order (previous, current) for CONTROL file order
    seen = []
    for f in [previous, current]:
        if f not in seen:
            seen.append(f)
    return seen


def ensure_downloaded(filename):
    """Download the GDAS file via FTP if not already present. Returns local path."""
    local_path = os.path.join(MET_DIR, filename)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"Already have {filename}")
        return local_path

    os.makedirs(MET_DIR, exist_ok=True)
    year = 2000 + int(filename.split(".")[1][-2:])
    url = f"{FTP_BASE}/{year}/{filename}"
    print(f"Fetching {url} ...")
    result = subprocess.run(
        ["curl", "-s", "-m", "600", "-o", local_path, url],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        raise RuntimeError(f"Failed to fetch {filename}: {result.stderr}")
    print(f"Downloaded {filename} ({os.path.getsize(local_path)} bytes)")
    return local_path


def ensure_met_files_for_event(event_dt):
    """Main entry point: guarantees both needed GDAS files are downloaded, returns their filenames."""
    needed = files_needed_for_event(event_dt)
    for f in needed:
        ensure_downloaded(f)
    return needed


if __name__ == "__main__":
    # Sanity check against the known event
    test_dt = dt.datetime(2026, 7, 19, 5, 0)
    files = files_needed_for_event(test_dt)
    print("Files needed for 2026-07-19T05:00 UTC event:", files)
    assert files == ["gdas1.jul26.w2", "gdas1.jul26.w3"], f"Mismatch: {files}"
    print("OK - matches what was manually fetched earlier")
