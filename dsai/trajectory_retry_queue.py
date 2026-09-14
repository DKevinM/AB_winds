# src/dsai/trajectory_retry_queue.py
#
# Persistent queue for HYSPLIT trajectory runs that failed because the
# needed met data genuinely didn't exist yet (see run_hysplit.py's
# 2026-09-13 fix - near-real-time gfsa cycles routinely lag ~9h behind
# wall-clock, so a same-hour trigger's start time can fall in a real
# gap). Kevin's call: retry at 01:00 the day after the original event,
# and if that's still short, once more at 01:00 the day after that -
# two delayed checks, then give up.
#
# Real bug found and fixed 2026-09-14: due_entries() originally compared
# calendar DATES only (first_failed_at.date() + N days), not elapsed
# time. That's fine for something that first fails early in the UTC
# day, but anything failing late in the day (which is most hourly
# checks, since failures happen all day) got its "next day" checkpoint
# only a few hours later - one real entry (Smoky Heights, failed 23:12,
# "retried" 01:00 the next calendar date) got retried after just 1h47m,
# nowhere near the ~9h lag, guaranteed to fail again. Confirmed this
# happened to most of an overnight queue of ~75 entries. Fixed with an
# explicit elapsed-hours floor (MIN_RETRY_ELAPSED_HOURS) on top of the
# calendar checkpoint - an entry not due to real time yet just waits
# for the following day's checkpoint instead, without being attempted
# (and without incrementing retries_attempted) prematurely.

import json
import os
import datetime as dt

QUEUE_PATH = "/opt/airquality/dsai_data/trajectory_retry_queue.json"
MAX_RETRIES = 2
# Just above the ~9h real-world lag observed 2026-09-13 (freshest
# published met data was still 9h stale at check time), not a large
# multiple of it - a bigger margin (e.g. 20h) would mean anything
# failing after ~05:00 UTC can't clear it by the very next 01:00
# checkpoint (a once-daily cron), pushing most of the day's failures
# out to a second checkpoint ~2 days later instead of Kevin's intended
# "next day." 12h is the smallest floor the real evidence supports -
# still real protection against the 1h47m case that motivated this
# fix, while keeping "next day" true for anything failing before ~13:00
# UTC (which is most of the watch window - AB/SK daytime hours).
MIN_RETRY_ELAPSED_HOURS = 12


def load_queue():
    if not os.path.exists(QUEUE_PATH):
        return []
    with open(QUEUE_PATH) as f:
        return json.load(f)


def save_queue(entries):
    os.makedirs(os.path.dirname(QUEUE_PATH), exist_ok=True)
    with open(QUEUE_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def enqueue(station, parameter, event_dt, duration_hours, cur_ts, result=None, now=None):
    """Add a failed run for delayed retry. event_dt: naive UTC datetime
    of the flagged reading. cur_ts: the original EXCEEDANCE log's
    timestamp string, kept only so a retry's log line can match it.
    result: the original check_exceedance() dict, kept so a successful
    retry can still tell whether the reading was extreme (see
    climatology.is_extreme) and queue the dual-model comparison the
    same as an immediate success would have."""
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
        "result": result,
        "first_failed_at": now.isoformat(),
        "retries_attempted": 0,
    })
    save_queue(entries)


def due_entries(now=None):
    """Entries whose next retry date has arrived - retry N is due
    starting at (first_failed_at.date() + N days), any time that day
    (the daily cron runs once, at 01:00) - AND at least
    MIN_RETRY_ELAPSED_HOURS have genuinely passed since that entry's
    last failure (first attempt for retry 1, the retry-1 failure for
    retry 2). An entry that clears the calendar checkpoint but not the
    elapsed-hours floor isn't attempted yet - it waits for the
    following day's checkpoint instead."""
    now = now or dt.datetime.now(dt.timezone.utc)
    today = now.date()
    entries = load_queue()
    due = []
    for e in entries:
        first_failed = dt.datetime.fromisoformat(e["first_failed_at"])
        due_date = first_failed.date() + dt.timedelta(days=e["retries_attempted"] + 1)
        if today < due_date:
            continue
        reference = dt.datetime.fromisoformat(e.get("last_failed_at") or e["first_failed_at"])
        elapsed_hours = (now - reference).total_seconds() / 3600.0
        if elapsed_hours < MIN_RETRY_ELAPSED_HOURS:
            continue
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
