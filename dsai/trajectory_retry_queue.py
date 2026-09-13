# src/dsai/trajectory_retry_queue.py
#
# Persistent queue for HYSPLIT trajectory runs that failed because the
# needed met data genuinely didn't exist yet (see run_hysplit.py's
# 2026-09-13 fix - near-real-time gfsa cycles routinely lag ~9h behind
# wall-clock, so a same-hour trigger's start time can fall in a real
# gap). Kevin's call: retry at 01:00 the day after the original event,
# and if that's still short, once more at 01:00 the day after that -
# two delayed checks, then give up. By the first retry, the needed
# hour is 9-33h old - comfortably past the observed ~9h lag - so this
# should resolve almost everything on the first retry; the second is a
# backstop, not the expected case.

import json
import os
import datetime as dt

QUEUE_PATH = "/opt/airquality/dsai_data/trajectory_retry_queue.json"
MAX_RETRIES = 2


def load_queue():
    if not os.path.exists(QUEUE_PATH):
        return []
    with open(QUEUE_PATH) as f:
        return json.load(f)


def save_queue(entries):
    os.makedirs(os.path.dirname(QUEUE_PATH), exist_ok=True)
    with open(QUEUE_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def enqueue(station, parameter, event_dt, duration_hours, cur_ts, now=None):
    """Add a failed run for delayed retry. event_dt: naive UTC datetime
    of the flagged reading. cur_ts: the original EXCEEDANCE log's
    timestamp string, kept only so a retry's log line can match it."""
    now = now or dt.datetime.now(dt.timezone.utc)
    entries = load_queue()
    key = (station, event_dt.isoformat(), duration_hours)
    if any((e["station"], e["event_dt"], e["duration_hours"]) == key for e in entries):
        return  # already queued - the hourly check re-fires on the same reading until new data lands
    entries.append({
        "station": station,
        "parameter": parameter,
        "event_dt": event_dt.isoformat(),
        "duration_hours": duration_hours,
        "cur_ts": cur_ts,
        "first_failed_at": now.isoformat(),
        "retries_attempted": 0,
    })
    save_queue(entries)


def due_entries(now=None):
    """Entries whose next retry date has arrived - retry N is due
    starting at (first_failed_at.date() + N days), any time that day
    (the daily cron runs once, at 01:00)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    today = now.date()
    entries = load_queue()
    due = []
    for e in entries:
        first_failed = dt.datetime.fromisoformat(e["first_failed_at"])
        due_date = first_failed.date() + dt.timedelta(days=e["retries_attempted"] + 1)
        if today >= due_date:
            due.append(e)
    return due


def mark_result(entry, succeeded, now=None):
    """succeeded=True removes the entry. succeeded=False either bumps
    retries_attempted (if under MAX_RETRIES) or removes it as given up."""
    now = now or dt.datetime.now(dt.timezone.utc)
    entries = load_queue()
    key = (entry["station"], entry["event_dt"], entry["duration_hours"])
    remaining = []
    gave_up = False
    for e in entries:
        if (e["station"], e["event_dt"], e["duration_hours"]) != key:
            remaining.append(e)
            continue
        if succeeded:
            continue  # drop - resolved
        e["retries_attempted"] += 1
        e["last_failed_at"] = now.isoformat()
        if e["retries_attempted"] >= MAX_RETRIES:
            gave_up = True
            continue  # drop - out of retries
        remaining.append(e)
    save_queue(remaining)
    return gave_up
