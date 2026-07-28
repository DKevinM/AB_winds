#!/bin/bash
set -e

set -a
source /opt/airquality/config/intelligence.env
set +a

cd /opt/airquality/github/AB_winds

/opt/airquality/venv/bin/python py/ab_met_pull.py --backfill 24
