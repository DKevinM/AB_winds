# src/dsai/retry_pending_trs.py
#
# Runs once a month (see run_dsai_retry_trs.sh / crontab). TRS's
# equivalent of retry_pending_trajectories.py, but for trs_retry_queue.py
# instead of the daily trajectory_retry_queue.py - see that module for
# why TRS gets its own, slower-cadence queue.
#
# Same "skip fire-context/downwind-receptors, they'd be mis-dated"
# reasoning as the daily retry script - a month-old TRS trigger's
# current wind reading tells you nothing about conditions at the time
# it fired.

import datetime as dt

from run_hysplit import run_ensemble
from climatology import is_extreme
import trs_retry_queue as queue
import dual_model_queue


def main():
    now = dt.datetime.now(dt.timezone.utc)
    due = queue.due_entries(now=now)
    if not due:
        print("No TRS retries due.")
        return

    print(f"{len(due)} TRS retr{'y' if len(due) == 1 else 'ies'} due.")
    for entry in due:
        station = entry["station"]
        parameter = entry["parameter"]
        event_dt = dt.datetime.fromisoformat(entry["event_dt"])
        duration_hours = entry["duration_hours"]
        attempt_n = entry["retries_attempted"] + 1

        print(f"TRS RETRY {attempt_n}/{queue.MAX_RETRIES}: {station} / {parameter} @ {entry['cur_ts']}")
        try:
            results, work_dir = run_ensemble(station, event_dt, duration_hours=duration_hours)
            succeeded = any(v is not None for v in results.values())
        except Exception as ex:
            print(f"  HYSPLIT run failed: {ex}")
            succeeded = False

        if succeeded:
            print(f"  HYSPLIT ensemble triggered for {station} @ {entry['cur_ts']} (TRS monthly retry {attempt_n})")
            if entry.get("result") and is_extreme(entry["result"]):
                print(f"  Extreme reading - queued for dual-model (HYSPLIT vs HRDPS) comparison")
                dual_model_queue.enqueue(station, parameter, event_dt, duration_hours, entry["cur_ts"], work_dir, now=now)
            queue.mark_result(entry, succeeded=True, now=now)
        else:
            gave_up = queue.mark_result(entry, succeeded=False, now=now)
            if gave_up:
                print(f"  Still no usable trajectory after {queue.MAX_RETRIES} monthly attempts - giving up on {station} @ {entry['cur_ts']}")
            else:
                print(f"  Still no usable trajectory - will retry again next month")


if __name__ == "__main__":
    main()
