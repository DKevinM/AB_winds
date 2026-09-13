# src/dsai/dual_model_queue.py
#
# Queue for extreme (see climatology.is_extreme) exceedances that
# should get the dual-model comparison (compare_trajectories.py)
# alongside their normal HYSPLIT trigger. Decoupled from the hourly
# check on purpose - a real timed run showed HRDPS takes ~7 minutes for
# a 24h trajectory (run_hrdps.py), which is too slow to run inline in
# check_exceedances.py without risking it overrunning into the next
# hourly tick (or, worse, the flock in run_dsai_watch.sh silently
# skipping that next run because the previous one is still holding the
# lock). A separate job (run_dsai_dual_model.sh, every 20 min) drains
# this queue instead - HYSPLIT has already succeeded and its work_dir
# is stored here, so draining only has to pay HRDPS's cost, not
# HYSPLIT's too.

import json
import os
import datetime as dt

QUEUE_PATH = "/opt/airquality/dsai_data/dual_model_queue.json"


def load_queue():
    if not os.path.exists(QUEUE_PATH):
        return []
    with open(QUEUE_PATH) as f:
        return json.load(f)


def save_queue(entries):
    os.makedirs(os.path.dirname(QUEUE_PATH), exist_ok=True)
    with open(QUEUE_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def enqueue(station, parameter, event_dt, duration_hours, cur_ts, hysplit_work_dir, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    entries = load_queue()
    key = (station, event_dt.isoformat(), duration_hours)
    if any((e["station"], e["event_dt"], e["duration_hours"]) == key for e in entries):
        return
    entries.append({
        "station": station,
        "parameter": parameter,
        "event_dt": event_dt.isoformat(),
        "duration_hours": duration_hours,
        "cur_ts": cur_ts,
        "hysplit_work_dir": hysplit_work_dir,
        "queued_at": now.isoformat(),
    })
    save_queue(entries)


def pop_all():
    """Returns every queued entry and empties the queue - the
    processing job handles all of them in one pass rather than a
    retry/backoff scheme, since extreme events are rare enough that the
    queue is almost always 0-1 entries deep."""
    entries = load_queue()
    save_queue([])
    return entries
