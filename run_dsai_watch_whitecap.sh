#!/bin/bash
set -e

# Daily, not hourly like run_dsai_watch.sh - new Whitecap incidents show
# up on whitecap_status_map's own weekly SK data refresh, and even a
# same-day miss just means a one-day-later trajectory, not a missed
# safety response (SK's incident feed itself already lags real
# occurrence by days). See dsai/watch_whitecap_incidents.py.

set -a
source /opt/airquality/config/intelligence.env
set +a

cd /opt/airquality/github/AB_winds/dsai

LOCKFILE="/opt/airquality/locks/dsai_watch_whitecap.lock"
mkdir -p "$(dirname "$LOCKFILE")"

(
  flock -n 200 || { echo "Previous dsai_watch_whitecap run still active; skipping."; exit 0; }
  /opt/airquality/venv/bin/python watch_whitecap_incidents.py
) 200>"$LOCKFILE"
