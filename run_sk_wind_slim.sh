#!/bin/bash
# SK_Air_Map animated winds: slim HRDPS files + hourly station adjustment (py/sk_wind_slim.py).
# Also the Sturgeon Lake dashboard's crop (region 'sturgeon', added 2026-10-27).
# Called hourly by cron and at the end of run_ab_met_pull.sh; flock keeps the two from overlapping.
set -a
. /opt/airquality/config/intelligence.env
set +a
cd /opt/airquality/github/AB_winds
exec 9>/opt/airquality/locks/sk_wind_slim.lock
flock -w 600 9 || { echo "$(date -u -Iseconds) sk_wind_slim: lock busy, skipped"; exit 0; }
/opt/airquality/venv/bin/python py/sk_wind_slim.py || echo "$(date -u -Iseconds) ALERT sk_wind_slim: SK animated-wind files not updated (see logs/sk_wind_slim.log)" >> /opt/airquality/logs/sitrep_alerts.log
/opt/airquality/venv/bin/python py/sk_wind_slim.py sturgeon || echo "$(date -u -Iseconds) ALERT sturgeon_wind_slim: Sturgeon Lake animated-wind files not updated (see logs/sk_wind_slim.log)" >> /opt/airquality/logs/sitrep_alerts.log
