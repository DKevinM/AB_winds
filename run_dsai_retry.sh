#!/bin/bash
set -e

# Daily at 01:00 UTC - picks up trajectory runs that failed because the
# needed met data wasn't published yet (see run_hysplit.py's 2026-09-13
# fix and trajectory_retry_queue.py). Two attempts, one day apart, then
# gives up - Kevin's call.

set -a
source /opt/airquality/config/intelligence.env
set +a

cd /opt/airquality/github/AB_winds/dsai

LOCKFILE="/opt/airquality/locks/dsai_retry.lock"
mkdir -p "$(dirname "$LOCKFILE")"

(
  flock -n 200 || { echo "Previous dsai_retry run still active; skipping."; exit 0; }
  /opt/airquality/venv/bin/python retry_pending_trajectories.py
) 200>"$LOCKFILE"
