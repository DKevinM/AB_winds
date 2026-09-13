# src/dsai/retry_pending_trajectories.py
#
# Runs once daily at 01:00 UTC (see run_dsai_retry.sh / crontab). Picks
# up anything trajectory_retry_queue.py queued because the met data
# wasn't published yet at trigger time, and tries again now that it's
# had 9-33h to show up. Two attempts total (see
# trajectory_retry_queue.MAX_RETRIES) before giving up and logging it
# as genuinely unresolved.
#
# Deliberately doesn't re-run the fire-context/downwind-receptor checks
# check_exceedances.py does for a live trigger - those use the LATEST
# wind reading, which is only correct for a live event, not a
# retry of something that happened a day or two ago. Re-running just
# the trajectory (the actual thing that was broken) and leaving
# fire/receptor context out rather than silently mis-dating it.

import datetime as dt

from run_hysplit import run_ensemble
from climatology import is_extreme
import trajectory_retry_queue as queue
import dual_model_queue


def main():
    now = dt.datetime.now(dt.timezone.utc)
    due = queue.due_entries(now=now)
    if not due:
        print("No trajectory retries due.")
        return

    print(f"{len(due)} trajectory retr{'y' if len(due) == 1 else 'ies'} due.")
    for entry in due:
        station = entry["station"]
        parameter = entry["parameter"]
        event_dt = dt.datetime.fromisoformat(entry["event_dt"])
        duration_hours = entry["duration_hours"]
        attempt_n = entry["retries_attempted"] + 1

        print(f"RETRY {attempt_n}/{queue.MAX_RETRIES}: {station} / {parameter} @ {entry['cur_ts']}")
        try:
            results, work_dir = run_ensemble(station, event_dt, duration_hours=duration_hours)
            succeeded = any(v is not None for v in results.values())
        except Exception as ex:
            print(f"  HYSPLIT run failed: {ex}")
            succeeded = False

        if succeeded:
            print(f"  HYSPLIT ensemble triggered for {station} @ {entry['cur_ts']} (delayed retry {attempt_n})")
            # entry["result"] is missing on anything queued before this
            # field existed (older queue entries) - .get() rather than
            # crash on those, just skip the dual-model check for them.
            if entry.get("result") and is_extreme(entry["result"]):
                print(f"  Extreme reading - queued for dual-model (HYSPLIT vs HRDPS) comparison")
                dual_model_queue.enqueue(station, parameter, event_dt, duration_hours, entry["cur_ts"], work_dir, now=now)
            queue.mark_result(entry, succeeded=True, now=now)
        else:
            gave_up = queue.mark_result(entry, succeeded=False, now=now)
            if gave_up:
                print(f"  Still no usable trajectory after {queue.MAX_RETRIES} attempts - giving up on {station} @ {entry['cur_ts']}")
            else:
                print(f"  Still no usable trajectory - will retry again tomorrow at 01:00")


if __name__ == "__main__":
    main()
