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
# Same western-Canada bbox as the LiveMap "ECCC Alerts" layer (BC/AB/SK/
# MB/YT/NT/NU), same VIIRS_SNPP_NRT source and 0.30 min-confidence floor
# as fire_hotspots.py's check_hotspots(), ported rather than shared since
# the two scripts query fundamentally different FIRMS endpoints
# (area-by-point-radius vs area-by-bbox).
#
# FLARE SUPPRESSION (added 2026-09-15): VIIRS active-fire detection can't
# tell a wildfire/prescribed burn from an oil & gas flare - both are just
# a sustained thermal anomaly to the sensor (this is exactly why NOAA
# built a *separate* VIIRS Nightfire product to catalog flares from this
# same raw data). Alberta has thousands of active flares, so this isn't
# a rare edge case for this bbox. Real fire-vs-flare literature leans on
# persistence + spectral signature; without spectral bands here, this
# uses the practical signal actually available: a flare sits at the same
# spot burning for weeks/months, a wildfire's footprint moves or dies
# out. `firms_flare_history.json` tracks, per ~1.1km grid cell, which
# calendar days (of the last HISTORY_RETENTION_DAYS) had a qualifying
# detection there. A cell hit on PERSISTENCE_THRESHOLD_DAYS+ of the last
# PERSISTENCE_WINDOW_DAYS is classified a likely flare and excluded from
# the wildfire-facing `features` (moved to `flares` instead, kept out of
# the map layer but counted so LiveMap's heads-up banner can say how many
# were filtered). Ramp-up caveat: with no history yet, nothing clears the
# threshold for the first ~PERSISTENCE_THRESHOLD_DAYS days after this
# shipped - suppression gets more accurate as history accumulates, it
# doesn't work retroactively on day one.

import json
import os
import sys
import csv
import io
from datetime import datetime, timedelta, timezone
import requests

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov"
WESTERN_CANADA_BBOX = "-141,48.2,-88,78"  # west,south,east,north
# 2026-11-20: VIIRS_SNPP_NRT has returned no rows since ~2026-11-02, so this layer (SNPP-only) went empty
# and the flare history froze. Now NOAA-20 + NOAA-21 (+ SNPP, in case it comes back). Persistence counts
# DAYS per cell, so more satellites don't change the threshold, just fill cloudy-day gaps.
SOURCES = ("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT")
BACKFILL_DAYS = 15   # --backfill: rebuild history from FIRMS (5-day windows, the API's day_range limit)
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

GRID_DECIMALS = 2  # ~1.1km grid cell for "same spot" matching
PERSISTENCE_WINDOW_DAYS = 14
PERSISTENCE_THRESHOLD_DAYS = 5
HISTORY_RETENTION_DAYS = 30  # kept a bit beyond the window for headroom/inspection

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "firms_hotspots.geojson")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "data", "firms_flare_history.json")


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


def _grid_key(lat, lon):
    return f"{round(lat, GRID_DECIMALS)}_{round(lon, GRID_DECIMALS)}"


def _load_history():
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _prune_history(history, today):
    cutoff = (today - timedelta(days=HISTORY_RETENTION_DAYS)).isoformat()
    pruned = {}
    for key, dates in history.items():
        kept = sorted(d for d in dates if d >= cutoff)
        if kept:
            pruned[key] = kept
    return pruned


def _persistence_days(history, key, today):
    window_cutoff = (today - timedelta(days=PERSISTENCE_WINDOW_DAYS)).isoformat()
    return sum(1 for d in history.get(key, []) if d >= window_cutoff)


def main():
    key = os.environ.get("FIRMS_API_KEY")
    if not key:
        print("FIRMS_API_KEY not set - skipping", file=sys.stderr)
        sys.exit(0)

    today = datetime.now(timezone.utc).date()
    rows, ok = [], 0
    for src in SOURCES:
        try:
            r = requests.get(f"{FIRMS_BASE_URL}/api/area/csv/{key}/{src}/{WESTERN_CANADA_BBOX}/{DAY_RANGE}", timeout=90)
            r.raise_for_status()
        except requests.RequestException as ex:
            print(f"FIRMS {src}: {ex}", file=sys.stderr)
            continue
        ok += 1
        got = list(csv.DictReader(io.StringIO(r.text)))
        for row in got:
            row["source"] = src
        rows += got
        print(f"FIRMS {src}: {len(got)} rows")
    if not ok:
        raise SystemExit("FIRMS: every source failed - keeping the previous file")

    history = _load_history()
    if "--backfill" in sys.argv:
        for src in SOURCES:
            for back in range(BACKFILL_DAYS, 0, -5):      # FIRMS: the date is the window START (date .. date+4)
                end = (today - timedelta(days=back)).isoformat()
                try:
                    r = requests.get(f"{FIRMS_BASE_URL}/api/area/csv/{key}/{src}/{WESTERN_CANADA_BBOX}/5/{end}", timeout=120)
                    r.raise_for_status()
                except requests.RequestException as ex:
                    print(f"backfill {src} {end}: {ex}", file=sys.stderr)
                    continue
                n = 0
                for row in csv.DictReader(io.StringIO(r.text)):
                    lat, lon = _num(row.get("latitude")), _num(row.get("longitude"))
                    cv = _confidence_value(row.get("confidence"))
                    if lat is None or lon is None or (cv is not None and cv < MIN_CONFIDENCE) or not row.get("acq_date"):
                        continue
                    history.setdefault(_grid_key(lat, lon), [])
                    if row["acq_date"] not in history[_grid_key(lat, lon)]:
                        history[_grid_key(lat, lon)].append(row["acq_date"])
                    n += 1
                print(f"backfill {src} from {end}: {n} detections")

    candidates = []
    for row in rows:
        lat, lon = _num(row.get("latitude")), _num(row.get("longitude"))
        if lat is None or lon is None:
            continue
        cv = _confidence_value(row.get("confidence"))
        if cv is not None and cv < MIN_CONFIDENCE:
            continue
        acq_date = row.get("acq_date") or today.isoformat()
        candidates.append((lat, lon, acq_date, row))

    # Record today's qualifying detections into history before classifying,
    # so a location's very first-ever detection still counts toward its
    # own persistence tally going forward.
    for lat, lon, acq_date, _row in candidates:
        gkey = _grid_key(lat, lon)
        history.setdefault(gkey, [])
        if acq_date not in history[gkey]:
            history[gkey].append(acq_date)

    history = _prune_history(history, today)

    features = []
    flares = []
    for lat, lon, acq_date, row in candidates:
        gkey = _grid_key(lat, lon)
        props = {
            "frp": _num(row.get("frp")),
            "confidence": row.get("confidence"),
            "acq_date": row.get("acq_date"),
            "acq_time": row.get("acq_time"),
            "daynight": row.get("daynight"),
            "satellite": row.get("source", "").replace("VIIRS_", "").replace("_NRT", ""),
        }
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props,
        }
        if _persistence_days(history, gkey, today) >= PERSISTENCE_THRESHOLD_DAYS:
            flares.append(feature)
        else:
            features.append(feature)

    fc = {
        "type": "FeatureCollection",
        "features": features,
        "flares": {"type": "FeatureCollection", "features": flares},
        "flare_count": len(flares),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(fc, f)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

    print(f"FIRMS hotspots: wrote {len(features)} detections ({len(flares)} likely flares filtered) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
