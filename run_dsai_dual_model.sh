#!/bin/bash
set -e

# Every 20 minutes - drains dual_model_queue.py (extreme exceedances
# where HYSPLIT succeeded, queued for the slower ~7min HRDPS
# comparison run rather than blocking the hourly check_exceedances.py
# pass). See run_dual_model_comparison.py.

set -a
source /opt/airquality/config/intelligence.env
set +a

cd /opt/airquality/github/AB_winds/dsai

LOCKFILE="/opt/airquality/locks/dsai_dual_model.lock"
mkdir -p "$(dirname "$LOCKFILE")"

(
  flock -n 200 || { echo "Previous dsai_dual_model run still active; skipping."; exit 0; }
  /opt/airquality/venv/bin/python run_dual_model_comparison.py
) 200>"$LOCKFILE"
