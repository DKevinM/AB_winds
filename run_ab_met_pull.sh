#!/bin/bash
set -e

set -a
source /opt/airquality/config/intelligence.env
set +a

cd /opt/airquality/github/AB_winds

/opt/airquality/venv/bin/python py/ab_met_pull.py
/opt/airquality/venv/bin/python py/cleanup_wind_files.py

# Small SK 10 m wind files for SK_Air_Map's animated wind layer (see py/sk_wind_slim.py).
# Its own failure must not look like a failed met pull, so it alerts separately.
/opt/airquality/venv/bin/python py/sk_wind_slim.py || echo "$(date -u -Iseconds) ALERT sk_wind_slim: SK animated-wind files not updated (see logs/ab_met_pull.log)" >> /opt/airquality/logs/sitrep_alerts.log
