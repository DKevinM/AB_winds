# wind_obs_nudge.py
#
# Station adjustment ("nudging") of model 10 m winds toward measured winds -
# the province-scale half of the Hinton / Howe Sound method (drafts/*_winds/
# dsmodel.py step 3), without their terrain model: on the prairies plain
# distance stands in for their valley distance.
#
# Observations: ECCC MSC SWOB real-time feed (api.weather.gc.ca swob-realtime),
# 10 m wind from MSC, NAV CANADA, SK-SPSA and other partner stations - roughly
# 200 stations an hour over SK and its margin, real-time with ~a week of history.
# (The climate-hourly archive the georeference check uses lags more than a day,
# so it can't drive a live map.)
#
# Correction (a successive-correction / Barnes step in u,v):
#   r_i  = obs_i - model_i               residual vector at each station
#   w_i  = exp(-(d_i / R)^2)             d = great-circle distance
#   corr = sum(w_i r_i) / max(sum(w_i), 1)
# Near several stations corr is their weighted mean residual; away from all
# stations (sum w < 1) it fades back to the plain model. R and the forecast
# fade time come from leave-one-station-out validation (validate_sk_nudge.py).

import csv
import math
import time
from pathlib import Path

import numpy as np
import requests

SWOB_URL = "https://api.weather.gc.ca/collections/swob-realtime/items"
PROPS = ("stn_nam-value,msc_id-value,data_pvdr-value,date_tm-value,"
         "avg_wnd_spd_10m_pst1hr,avg_wnd_dir_10m_pst1hr,avg_wnd_spd_10m_pst10mts,avg_wnd_dir_10m_pst10mts")
EARTH_KM = 6371.0
EXCLUDE_CSV = Path(__file__).resolve().parent.parent / "data" / "wind_station_exclude.csv"


def excluded_names():
    """Stations with known-bad wind (data/wind_station_exclude.csv, by SWOB station name)."""
    try:
        with open(EXCLUDE_CSV, newline="") as f:
            return {r["name"].strip().upper() for r in csv.DictReader(f) if r.get("name")}
    except FileNotFoundError:
        return set()


def fetch_swob(hour_utc, bbox, retries=3):
    """10 m winds observed at one top-of-hour time. Returns a list of dicts
    {name, id, lat, lon, u, v, spd} with u/v in m/s (earth-relative, toward), spd in m/s.
    One record per station; the 1-hour mean is preferred over the 10-minute mean."""
    params = {"f": "json", "limit": 2000, "bbox": ",".join(str(x) for x in bbox),
              "datetime": hour_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "properties": PROPS}
    for attempt in range(retries):
        try:
            r = requests.get(SWOB_URL, params=params, timeout=90)
            r.raise_for_status()
            feats = r.json().get("features", [])
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    out, skip = {}, excluded_names()
    for ft in feats:
        p = ft["properties"]
        if (p.get("stn_nam-value") or "").strip().upper() in skip:
            continue
        spd, d = p.get("avg_wnd_spd_10m_pst1hr"), p.get("avg_wnd_dir_10m_pst1hr")
        if spd is None or d is None:
            spd, d = p.get("avg_wnd_spd_10m_pst10mts"), p.get("avg_wnd_dir_10m_pst10mts")
        try:
            spd, d = float(spd), float(d)
        except (TypeError, ValueError):
            continue
        if not (0 <= spd <= 150 and 0 <= d <= 360):
            continue
        lon, lat = ft["geometry"]["coordinates"][:2]
        sid = p.get("msc_id-value") or p.get("stn_nam-value")
        ms = spd / 3.6
        # meteorological direction = where the wind comes FROM; u/v point where it goes
        u, v = -ms * math.sin(math.radians(d)), -ms * math.cos(math.radians(d))
        if spd == 0:
            u = v = 0.0
        out[sid] = {"name": p.get("stn_nam-value"), "id": sid, "lat": float(lat), "lon": float(lon),
                    "u": u, "v": v, "spd": ms}
    return list(out.values())


def gc_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km); broadcasts over numpy arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def correction_at(st_lat, st_lon, res_u, res_v, lat, lon, radius_km):
    """Correction vector at target points (lat/lon arrays of any shape) from station residuals."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    cu, cv, wsum = np.zeros(lat.shape), np.zeros(lat.shape), np.zeros(lat.shape)
    for la, lo, ru, rv in zip(st_lat, st_lon, res_u, res_v):
        w = np.exp(-(gc_km(la, lo, lat, lon) / radius_km) ** 2)
        cu += w * ru; cv += w * rv; wsum += w
    den = np.maximum(wsum, 1.0)
    return cu / den, cv / den


def buddy_check(st_lat, st_lon, res_u, res_v, radius_km, floor_ms=5.0, k=3.0):
    """Mask of stations to keep: a station whose residual disagrees with what its
    neighbours' residuals predict there (leave-one-out) by more than
    max(floor_ms, k * median disagreement) is dropped for this hour - e.g. a vane
    reporting the opposite direction, or a sheltered/broken anemometer."""
    lat, lon = np.asarray(st_lat, float), np.asarray(st_lon, float)
    ru, rv = np.asarray(res_u, float), np.asarray(res_v, float)
    W = np.exp(-(gc_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) / radius_km) ** 2)
    np.fill_diagonal(W, 0.0)
    den = np.maximum(W.sum(1), 1.0)
    miss = np.hypot(ru - (W @ ru) / den, rv - (W @ rv) / den)
    return miss <= max(floor_ms, k * float(np.median(miss)))


def sample_grid(grid, U, V, lat, lon):
    """Bilinear sample of a north-up grid (i = (lon-lo1)/dx, j = (la1-lat)/dy); NaN outside."""
    fi = (np.asarray(lon, float) - grid["lo1"]) / grid["dx"]
    fj = (grid["la1"] - np.asarray(lat, float)) / grid["dy"]
    ok = (fi >= 0) & (fj >= 0) & (fi < grid["nx"] - 1) & (fj < grid["ny"] - 1)
    i = np.where(ok, fi, 0).astype(int); j = np.where(ok, fj, 0).astype(int)
    di, dj = np.where(ok, fi - i, 0), np.where(ok, fj - j, 0)
    bl = lambda A: (A[j, i] * (1 - di) * (1 - dj) + A[j, i + 1] * di * (1 - dj)
                    + A[j + 1, i] * (1 - di) * dj + A[j + 1, i + 1] * di * dj)
    u, v = bl(U), bl(V)
    return np.where(ok, u, np.nan), np.where(ok, v, np.nan)
