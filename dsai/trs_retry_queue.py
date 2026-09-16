# src/dsai/trs_retry_queue.py
#
# Kevin's call 2026-09-15: TRS was by far the largest share of the
# daily HYSPLIT retry backlog (31 of 51 entries the day this was
# built - more stations report TRS than H2S/SO2/PM2.5 combined) and
# it's the lowest-priority parameter of the four for DSAI's purposes
# (no Alberta AAQO of its own - see the 2026-09-14 Edmonton East H2S
# verification, where the TRS AAQO in play was one borrowed from Metro
# Vancouver). Rather than compete with H2S/SO2/PM2.5 for the daily
# 01:00 retry slot, TRS failures get parked here and only retried
# once a month instead - see retry_pending_trs.py / crontab's monthly
# run.
#
# Same enqueue/dedup/give-up shape as trajectory_retry_queue.py, but
# "due" means "a new calendar month has started since this entry's
# last failure" rather than a same-day elapsed-hours floor - there's
# no equivalent same-day urgency for TRS that MIN_RETRY_ELAPSED_HOURS
# was protecting against there.

import json
import os
import datetime as dt

QUEUE_PATH = "/opt/airquality/dsai_data/trs_retry_queue.json"
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


def enqueue(station, parameter, event_dt, duration_hours, cur_ts, result=None, now=None):
    """Same signature as trajectory_retry_queue.enqueue - see there for
    field meanings."""
    now = now or dt.datetime.now(dt.timezone.utc)
    entries = load_queue()
    key = (station, event_dt.isoformat(), duration_hours)
    if any((e["station"], e["event_dt"], e["duration_hours"]) == key for e in entries):
        return  # already queued
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


def _month_index(d):
    return d.year * 12 + d.month


def due_entries(now=None):
    """Due once the current calendar month is later than the month of
    this entry's last failure (first failure for retry 1, that retry's
    failure for retry 2) - i.e. the monthly cron run after an entry
    first failed always picks it up, same as trajectory_retry_queue's
    "next day" but at monthly grain."""
    now = now or dt.datetime.now(dt.timezone.utc)
    entries = load_queue()
    due = []
    for e in entries:
        reference = dt.datetime.fromisoformat(e.get("last_failed_at") or e["first_failed_at"])
        if _month_index(now.date()) > _month_index(reference.date()):
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
