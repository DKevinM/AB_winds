# src/dsai/gdas_fetch.py
#
# Met data for HYSPLIT, from two different NOAA ARL sources depending on
# how recent the event is:
#
#   - Permanent archive (gdas1, weekly files) - authoritative, but NOAA
#     doesn't post a week's file until ~1-2 days after that week ends.
#     Confirmed against the live archive listing, 2026-08-16:
#       W1 = days 1-7, W2 = 8-14, W3 = 15-21, W4 = 22-28, W5 = 29-end.
#
#   - Near-real-time forecast/analysis directory (gfsa, per-cycle files
#     at ftp.arl.noaa.gov/forecast/<YYYYMMDD>/) - published within about
#     an hour of each 00/06/12/18z cycle, but only kept online for about
#     a week before rolling off. Used when an event is too recent for
#     the permanent archive to have posted yet.

import os
import subprocess
import time
import datetime as dt

MET_DIR = "/opt/airquality/hysplit/met_data"
ARCHIVE_FTP_BASE = "ftp://arlftp.arlhq.noaa.gov/pub/archives/gdas1"
FORECAST_FTP_BASE = "ftp://ftp.arl.noaa.gov/forecast"

MONTH_ABBR = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
}

# Observed lag between a week ending and its archive file posting is
# 1-2 days; require 3 clear days as a safety margin before trusting it.
ARCHIVE_SAFE_LAG_DAYS = 3


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


def week_end_date(year, month, week):
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    ends = {1: 7, 2: 14, 3: 21, 4: 28, 5: last_day}
    return dt.date(year, month, ends[week])


def previous_week_file(year, month, week):
    start = week_start_date(year, month, week)
    prev_day = start - dt.timedelta(days=1)
    return gdas_filename(prev_day.year, prev_day.month, week_of_month(prev_day.day))


def archive_available(event_dt, now=None):
    """
    Whether the permanent weekly archive covering event_dt is safely
    expected to be posted by now.
    """
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    week = week_of_month(event_dt.day)
    end = week_end_date(event_dt.year, event_dt.month, week)
    safe_after = dt.datetime.combine(end, dt.time(0, 0)) + dt.timedelta(days=ARCHIVE_SAFE_LAG_DAYS + 1)
    return now >= safe_after


def files_needed_for_event(event_dt, duration_hours=72):
    """
    Which archive week-files are needed to cover a backward trajectory
    of duration_hours starting at event_dt. Includes a 12h safety
    margin around the week boundary for interpolation at file edges.
    """
    reach_back = event_dt - dt.timedelta(hours=duration_hours)
    week_event = week_of_month(event_dt.day)
    current = gdas_filename(event_dt.year, event_dt.month, week_event)

    week_start = dt.datetime.combine(
        week_start_date(event_dt.year, event_dt.month, week_event), dt.time(0, 0)
    )
    files = [current]
    if reach_back < week_start + dt.timedelta(hours=12):
        prev = previous_week_file(event_dt.year, event_dt.month, week_event)
        files.insert(0, prev)
    return files


def ensure_downloaded(filename, subdir="", retries=3, retry_delay_s=15):
    """Download with retries - a multi-thousand-request batch job WILL hit
    transient FTP hiccups eventually; one blip shouldn't kill the whole run
    (this is exactly what took down the first overnight Henry Pirker batch)."""
    local_path = os.path.join(MET_DIR, subdir, filename) if subdir else os.path.join(MET_DIR, filename)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"Already have {filename}")
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if subdir:
        # near-real-time forecast directory file
        url = f"{FORECAST_FTP_BASE}/{subdir}/{filename}"
    else:
        year = 2000 + int(filename.split(".")[1][-2:])
        url = f"{ARCHIVE_FTP_BASE}/{year}/{filename}"

    last_err = None
    for attempt in range(1, retries + 1):
        print(f"Fetching {url} (attempt {attempt}/{retries}) ...")
        result = subprocess.run(
            ["curl", "-s", "-m", "600", "-o", local_path, url],
            capture_output=True, text=True
        )
        if result.returncode == 0 and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            print(f"Downloaded {filename} ({os.path.getsize(local_path)} bytes)")
            return local_path

        last_err = result.stderr or f"curl exit code {result.returncode}"
        if os.path.exists(local_path):
            os.remove(local_path)
        if attempt < retries:
            print(f"  attempt {attempt} failed ({last_err}); retrying in {retry_delay_s}s")
            time.sleep(retry_delay_s)

    raise RuntimeError(f"Failed to fetch {filename} after {retries} attempts: {last_err}")


def recent_cycle_files_needed(event_dt, duration_hours):
    """
    Near-real-time fallback: gfsa cycle files (00/06/12/18z) covering
    the requested backward window, plus a one-day safety buffer, plus
    tomorrow's date in case event_dt is "now" and the clock ticks over
    a day boundary mid-run.
    """
    days_back = (duration_hours // 24) + 2
    files = []
    for d in range(-1, days_back):
        day = (event_dt - dt.timedelta(days=d)).strftime("%Y%m%d")
        for cycle in ["00", "06", "12", "18"]:
            files.append((day, f"hysplit.t{cycle}z.gfsa"))
    return files


def ensure_met_files_for_event(event_dt, duration_hours=72, now=None):
    """
    Main entry point. Returns (met_files, met_subdir) where met_subdir
    is "" for the permanent archive (files live directly in MET_DIR) or
    a per-file subdir map for the near-real-time fallback.
    Raises if neither source can supply what's needed.
    """
    if archive_available(event_dt, now=now):
        needed = files_needed_for_event(event_dt, duration_hours=duration_hours)
        for f in needed:
            ensure_downloaded(f)
        return [(f, "") for f in needed]

    print(f"Archive not yet available for {event_dt} - using near-real-time gfsa cycles instead")
    candidates = recent_cycle_files_needed(event_dt, duration_hours)
    fetched = []
    for day, fname in candidates:
        try:
            ensure_downloaded(fname, subdir=day)
            fetched.append((fname, day))
        except RuntimeError as ex:
            print(f"  skip {day}/{fname}: {ex}")
    if not fetched:
        raise RuntimeError(f"No near-real-time met data available for {event_dt}")
    return fetched


if __name__ == "__main__":
    # Sanity check against the known (now-archived) event, both durations
    test_dt = dt.datetime(2026, 7, 19, 5, 0)
    for dur in [24, 72]:
        files = files_needed_for_event(test_dt, duration_hours=dur)
        print(f"{dur}h duration needs: {files}")
    # 72h reaches back to Jul 16 05:00 - 17h past the Jul 15 week-3
    # boundary, outside the 12h margin, so w3 alone genuinely suffices.
    # (The original hand-built fetch grabbed w2 too, out of caution that
    # wasn't strictly necessary - the actual HYSPLIT run only used w3's
    # data for this trajectory either way.) 24h reaches back to Jul 18
    # 05:00, deep inside week 3 -> also just w3.
    assert files_needed_for_event(test_dt, duration_hours=72) == ["gdas1.jul26.w3"]
    assert files_needed_for_event(test_dt, duration_hours=24) == ["gdas1.jul26.w3"]
    # A trajectory starting right at a week boundary should pull both weeks
    edge_dt = dt.datetime(2026, 7, 15, 3, 0)  # 3h into week 3
    edge_files = files_needed_for_event(edge_dt, duration_hours=24)
    assert edge_files == ["gdas1.jul26.w2", "gdas1.jul26.w3"], edge_files

    # Month-boundary case: 3h into week 1 of August should reach back
    # into July's last week
    month_edge_dt = dt.datetime(2026, 8, 1, 3, 0)
    month_edge_files = files_needed_for_event(month_edge_dt, duration_hours=24)
    assert month_edge_files == ["gdas1.jul26.w5", "gdas1.aug26.w1"], month_edge_files
    print("OK")

    # Sanity check the near-real-time path selection for "right now"
    now_dt = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    print()
    print("archive_available(now) ->", archive_available(now_dt))
    print("recent_cycle_files_needed(now, 24) sample:", recent_cycle_files_needed(now_dt, 24)[:4], "...")
