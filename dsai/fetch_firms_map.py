# src/dsai/fetch_firms_map.py
#
# NASA FIRMS active-fire export for LiveMap's toggleable "Fire hotspots"
# layer. Distinct from fire_hotspots.py's check_hotspots(): that one
# answers "is there fire near this one station" for check_exceedances.py;
# this one exports every VIIRS detection across the whole western-Canada
# bbox as a static GeoJSON file, refreshed on a cron and served over
# plain HTTP - the FIRMS area API takes the MAP_KEY in the URL path, so
# it can't be called straight from LiveMap's public browser JS the way
# the ECCC alerts layer is (that endpoint needs no key at all). This
# script is the server side of that split: it holds the key, LiveMap's
# JS only ever sees the resulting GeoJSON.
#
# Same western-Canada bbox as the LiveMap "Environment Canada Alerts"
# layer (BC/AB/SK/MB/YT/NT/NU), same VIIRS_SNPP_NRT source and 0.30
# min-confidence floor as fire_hotspots.py's check_hotspots(), ported
# rather than shared since the two scripts query fundamentally different
# FIRMS endpoints (area-by-point-radius vs area-by-bbox).

import json
import os
import sys
import csv
import io
import requests

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov"
WESTERN_CANADA_BBOX = "-141,48.2,-88,78"  # west,south,east,north
SOURCE = "VIIRS_SNPP_NRT"
# day_range counts back from "today" in FIRMS' own clock, and a request
# with no explicit date defaults to today - which can have near-zero
# rows for the first few hours after UTC midnight, before that day's
# satellite passes have been processed (confirmed directly: 0 rows at
# 00:39 UTC with day_range=1, full rows with day_range=2 covering
# yesterday+today at the same moment). 2 keeps this layer populated
# through that daily gap instead of going falsely empty.
DAY_RANGE = 2
MIN_CONFIDENCE = 0.30
CONF_MAP = {"l": 0.3, "n": 0.6, "h": 0.9}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "firms_hotspots.geojson")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _confidence_value(v):
    if v in CONF_MAP:
        return CONF_MAP[v]
    n = _num(v)
    return n / 100 if n is not None else None


def main():
    key = os.environ.get("FIRMS_API_KEY")
    if not key:
        print("FIRMS_API_KEY not set - skipping", file=sys.stderr)
        sys.exit(0)

    url = f"{FIRMS_BASE_URL}/api/area/csv/{key}/{SOURCE}/{WESTERN_CANADA_BBOX}/{DAY_RANGE}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.text)))

    features = []
    for row in rows:
        lat, lon = _num(row.get("latitude")), _num(row.get("longitude"))
        if lat is None or lon is None:
            continue
        cv = _confidence_value(row.get("confidence"))
        if cv is not None and cv < MIN_CONFIDENCE:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "frp": _num(row.get("frp")),
                "confidence": row.get("confidence"),
                "acq_date": row.get("acq_date"),
                "acq_time": row.get("acq_time"),
                "daynight": row.get("daynight"),
            }
        })

    fc = {"type": "FeatureCollection", "features": features}

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(fc, f)

    print(f"FIRMS hotspots: wrote {len(features)} detections to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
