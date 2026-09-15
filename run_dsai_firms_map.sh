#!/bin/bash
set -e

set -a
source /opt/airquality/config/intelligence.env
set +a

cd /opt/airquality/github/AB_winds/dsai

LOCKFILE="/opt/airquality/locks/dsai_firms_map.lock"
mkdir -p "$(dirname "$LOCKFILE")"

(
  flock -n 200 || { echo "Previous dsai_firms_map run still active; skipping."; exit 0; }
  /opt/airquality/venv/bin/python fetch_firms_map.py
) 200>"$LOCKFILE"
