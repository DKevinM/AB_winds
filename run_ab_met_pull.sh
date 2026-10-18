#!/bin/bash
set -e

set -a
source /opt/airquality/config/intelligence.env
set +a

cd /opt/airquality/github/AB_winds

/opt/airquality/venv/bin/python py/ab_met_pull.py
/opt/airquality/venv/bin/python py/cleanup_wind_files.py

# Small SK 10 m wind files for SK_Air_Map's animated wind layer (see py/sk_wind_slim.py);
# also runs hourly from cron for the station adjustment. Alerts on its own failure.
/opt/airquality/github/AB_winds/run_sk_wind_slim.sh >> /opt/airquality/logs/sk_wind_slim.log 2>&1
