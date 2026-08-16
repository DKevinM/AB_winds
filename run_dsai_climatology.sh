#!/bin/bash
set -e

set -a
source /opt/airquality/config/intelligence.env
set +a

cd /opt/airquality/github/AB_winds/dsai

/opt/airquality/venv/bin/python build_climatology_cache.py
