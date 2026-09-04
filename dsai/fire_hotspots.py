# src/dsai/fire_hotspots.py
#
# NASA FIRMS active-fire check, ported from the sit-rep pipelines'
# modules/fire/service.py (riders_sitrep, edmonton_folk_fest, etc.) -
# same VIIRS_SNPP_NRT area-CSV endpoint, same confidence/clustering
# logic, adapted to DSAI's plain lat/lon calling convention instead of
# a cfg/event dict, and reusing facilities.bearing_and_distance()
# instead of a second haversine implementation.
#
# Wired into check_exceedances.py: every H2S/SO2/TRS/PM2.5 trigger also
# checks for nearby fire activity as likely context, the same way it
# already runs HYSPLIT for likely transport. Covers both wildfire and
# prescribed-burn detections - FIRMS doesn't distinguish the two, satellite
# just sees heat.

import os
import csv
import io
import math
import requests

from facilities import bearing_and_distance

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov"
CONF_MAP = {"l": 0.3, "n": 0.6, "h": 0.9}


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


def _compass(bearing):
    if bearing is None:
        return "unknown"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(bearing / 45) % 8]


def _bbox_for(lat, lon, radius_km):
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def check_hotspots(lat, lon, radius_km=150, cluster_km=10, min_confidence=0.30,
                    source="VIIRS_SNPP_NRT", day_range=1, timeout_seconds=20):
    """
    Nearby NASA FIRMS active-fire detections, clustered and sorted by
    distance. Returns {"status": "missing"|"error"|"ok", ...}.

    radius_km defaults to 150 (smaller than the sit-reps' 300) - these
    are fixed monitoring stations checked hourly rather than a single
    event site, so a tighter radius keeps "there's a fire nearby" a
    meaningful signal rather than picking up half the province during
    fire season.
    """
    key = os.environ.get("FIRMS_API_KEY")
    if not key:
        return {"status": "missing", "reason": "FIRMS_API_KEY not set in environment"}

    w, s, e, n = _bbox_for(lat, lon, radius_km)
    url = f"{FIRMS_BASE_URL}/api/area/csv/{key}/{source}/{w:.4f},{s:.4f},{e:.4f},{n:.4f}/{day_range}"

    try:
        r = requests.get(url, timeout=timeout_seconds)
        r.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(r.text)))
    except Exception as ex:
        return {"status": "error", "error": f"{type(ex).__name__}: {ex}"}

    candidates = []
    for row in rows:
        rlat, rlon = _num(row.get("latitude")), _num(row.get("longitude"))
        if rlat is None or rlon is None:
            continue
        cv = _confidence_value(row.get("confidence"))
        if cv is not None and cv < min_confidence:
            continue
        bearing, dist = bearing_and_distance(lat, lon, rlat, rlon)
        if dist > radius_km:
            continue
        candidates.append({
            "lat": rlat, "lon": rlon,
            "distance_km": round(dist, 1),
            "bearing_deg": round(bearing, 1),
            "direction": _compass(bearing),
            "frp": _num(row.get("frp")),
            "confidence": row.get("confidence"),
            "acq_date": row.get("acq_date"),
            "acq_time": row.get("acq_time"),
            "daynight": row.get("daynight"),
        })

    candidates.sort(key=lambda c: c["distance_km"])

    clustered = []
    for c in candidates:
        if any(bearing_and_distance(c["lat"], c["lon"], k["lat"], k["lon"])[1] <= cluster_km for k in clustered):
            continue
        clustered.append(c)

    if not clustered:
        return {"status": "ok", "count": 0, "hotspots": [], "nearest": None}

    return {"status": "ok", "count": len(clustered), "hotspots": clustered[:20], "nearest": clustered[0]}
