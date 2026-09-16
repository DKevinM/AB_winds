#!/bin/bash
set -e

# Monthly (1st of the month) - TRS's equivalent of run_dsai_retry.sh.
# Kevin's call 2026-09-15: TRS was the largest single share of the
# daily retry backlog and doesn't need same-day urgency (no Alberta
# AAQO of its own), so it gets parked and only retried once a month
# instead - see dsai/trs_retry_queue.py.

set -a
source /opt/airquality/config/intelligence.env
set +a

cd /opt/airquality/github/AB_winds/dsai

LOCKFILE="/opt/airquality/locks/dsai_retry_trs.lock"
mkdir -p "$(dirname "$LOCKFILE")"

(
  flock -n 200 || { echo "Previous dsai_retry_trs run still active; skipping."; exit 0; }
  /opt/airquality/venv/bin/python retry_pending_trs.py
) 200>"$LOCKFILE"
