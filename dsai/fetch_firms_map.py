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
#
# 2026-11-20: brought in line with the PAZA/WCAS emergency maps (paza_emergency_intelligence/flares.py),
# because the same-spot-only rule missed most flares (Nov 19: 4 of PAZA's 8 "fires" were small night-time
# heat next to gas facilities). History now keeps night/day per cell ({cell: {date: "N"|"D"|"DN"}}) and a
# detection is a likely flare when it is SMALL (neighbourhood peak < 10 MW, median < 5 MW) AND either
#   (a) mostly at night within 2 km of an active AER facility (data/flare_sites.json, AB only), or
#   (b) heat at the same spot (~1 km) on 3+ nights in the last 20 days;
# or, at any size,
#   (c) 5+ nights at the same spot AND within 2 km of a facility, or
#   (d) the original rule: heat at the same spot on 5+ of the last 14 days.
# Likely flares are no longer hidden: LiveMap draws them as grey dashed rings with the reason
# ("identify, don't remove"), and the heads-up banner counts only the fires.

import json
import os
import sys
import csv
import statistics
import io
from datetime import datetime, timedelta, timezone
import numpy as np
import requests

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov"
WESTERN_CANADA_BBOX = "-141,48.2,-88,78"  # west,south,east,north
# 2026-11-20: VIIRS_SNPP_NRT has returned no rows since ~2026-11-02, so this layer (SNPP-only) went empty
# and the flare history froze. Now NOAA-20 + NOAA-21 (+ SNPP, in case it comes back). Persistence counts
# DAYS per cell, so more satellites don't change the threshold, just fill cloudy-day gaps.
SOURCES = ("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT")
BACKFILL_DAYS = 20   # --backfill (or old/missing history): rebuild history from FIRMS (5-day windows, the API's day_range limit)
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

STEP = 0.01                   # history cell (deg); "same spot" = +/- 1 cell N-S, +/- 2 cells E-W (~1 km)
PERSISTENCE_WINDOW_DAYS = 14  # rule (d)
PERSISTENCE_THRESHOLD_DAYS = 5
MEDIAN_FRP = 5.0              # MW; flares in VIIRS are mostly 0.5-5 MW
MAX_FRP = 10.0                # MW
FAC_KM = 2.0
REPEAT_NIGHTS = 3             # rule (b)
PERSIST_NIGHTS = 5            # rule (c)
HISTORY_RETENTION_DAYS = 20   # also the rule (b)/(c) window

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "firms_hotspots.geojson")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "data", "firms_flare_history.json")
SITES_PATH = os.path.join(os.path.dirname(__file__), "data", "flare_sites.json")


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
    return f"{lat:.2f},{lon:.2f}"


def _add(history, lat, lon, date, night):
    c = history.setdefault(_grid_key(lat, lon), {})
    flag = "N" if night else "D"
    if flag not in c.get(date, ""):
        c[date] = "".join(sorted(c.get(date, "") + flag))


def _load_history():
    try:
        with open(HISTORY_PATH) as f:
            h = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if any(isinstance(v, list) for v in h.values()):   # pre-2026-11-20 format (dates only, no night flag)
        return None
    return h


def _prune_history(history, today):
    cutoff = (today - timedelta(days=HISTORY_RETENTION_DAYS)).isoformat()
    pruned = {}
    for key, dates in history.items():
        kept = {d: v for d, v in dates.items() if d >= cutoff}
        if kept:
            pruned[key] = kept
    return pruned


def _dates_near(history, points, night_only):
    """Distinct dates with heat (or night heat) within ~1 km of any of the points."""
    out = set()
    for lat, lon in points:
        for i in (-1, 0, 1):
            for j in (-2, -1, 0, 1, 2):
                c = history.get(_grid_key(lat + i * STEP, lon + j * STEP))
                if c:
                    out |= {d for d, v in c.items() if not night_only or "N" in v}
    return out


_sites = None


def _nearest_site(lat, lon):
    global _sites
    if _sites is None:
        try:
            s = json.load(open(SITES_PATH))["sites"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            s = []
        _sites = (np.array([[x[0], x[1]] for x in s]).reshape(-1, 2), s)
    xy, s = _sites
    if not len(s):
        return None
    d = np.hypot((xy[:, 1] - lon) * 111.32 * np.cos(np.radians(lat)), (xy[:, 0] - lat) * 111.32)
    i = int(d.argmin())
    if d[i] > 25:                      # outside Alberta / no facility anywhere near
        return None
    return {"name": s[i][2], "type": s[i][3], "km": round(float(d[i]), 1)}


def _classify(members, history, today):
    """members: today's detections within ~1 km of this one (incl. itself): dicts lat, lon, frp, night."""
    frps = [m["frp"] for m in members if m["frp"] is not None] or [0.0]
    lat = sum(m["lat"] for m in members) / len(members)
    lon = sum(m["lon"] for m in members) / len(members)
    site = _nearest_site(lat, lon)
    near = site is not None and site["km"] <= FAC_KM
    night_share = sum(m["night"] for m in members) / len(members)
    pts = {(round(m["lat"], 2), round(m["lon"], 2)) for m in members}
    nights = len(_dates_near(history, pts, True))
    win = (today - timedelta(days=PERSISTENCE_WINDOW_DAYS)).isoformat()
    days14 = len({d for d in _dates_near(history, pts, False) if d >= win})
    small = max(frps) < MAX_FRP and statistics.median(frps) < MEDIAN_FRP
    by_site = near and night_share >= 0.5
    by_repeat = nights >= REPEAT_NIGHTS
    persistent = near and nights >= PERSIST_NIGHTS
    by_days = days14 >= PERSISTENCE_THRESHOLD_DAYS
    flare = (small and (by_site or by_repeat)) or persistent or by_days
    why = []
    if flare:
        why.append(f"small heat (peak {max(frps):.1f} MW)" if small else f"peak {max(frps):.1f} MW")
        if small and by_site:
            why.append(f"{'at night' if night_share == 1 else 'mostly at night'}, "
                       f"{site['km']} km from {site['name']} ({site['type']})")
        elif near:
            why.append(f"{site['km']} km from {site['name']} ({site['type']})")
        if (small and by_repeat) or persistent:
            why.append(f"heat at this spot on {nights} nights in the last {HISTORY_RETENTION_DAYS} days")
        elif by_days:
            why.append(f"heat at this spot on {days14} of the last {PERSISTENCE_WINDOW_DAYS} days")
    return {"likely_flare": flare, "flare_reason": "; ".join(why), "nearest_facility": site,
            "nights_here": nights}


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

    try:                                   # monthly: active AER facilities for the facility rule
        import flare_sites
        flare_sites.refresh()
    except Exception as ex:                # never let this stop the hotspot layer
        print(f"flare sites refresh failed: {ex}", file=sys.stderr)

    history = _load_history()
    if history is None or "--backfill" in sys.argv:
        history = {}
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
                    _add(history, lat, lon, row["acq_date"], row.get("daynight") == "N")
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
        candidates.append({"lat": lat, "lon": lon, "date": acq_date, "row": row,
                           "frp": _num(row.get("frp")), "night": row.get("daynight") == "N"})

    # Record today's qualifying detections into history before classifying,
    # so a location's very first-ever detection still counts toward its
    # own persistence tally going forward.
    by_cell = {}
    for c in candidates:
        _add(history, c["lat"], c["lon"], c["date"], c["night"])
        by_cell.setdefault((round(c["lat"], 2), round(c["lon"], 2)), []).append(c)

    history = _prune_history(history, today)

    features = []
    flares = []
    for c in candidates:
        row = c["row"]
        clat, clon = round(c["lat"], 2), round(c["lon"], 2)
        members = [m for i in (-1, 0, 1) for j in (-2, -1, 0, 1, 2)
                   for m in by_cell.get((round(clat + i * STEP, 2), round(clon + j * STEP, 2)), [])]
        props = {
            "frp": c["frp"],
            "confidence": row.get("confidence"),
            "acq_date": row.get("acq_date"),
            "acq_time": row.get("acq_time"),
            "daynight": row.get("daynight"),
            "satellite": row.get("source", "").replace("VIIRS_", "").replace("_NRT", ""),
        }
        props.update(_classify(members, history, today))
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]},
            "properties": props,
        }
        (flares if props["likely_flare"] else features).append(feature)

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
