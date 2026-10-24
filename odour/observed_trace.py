"""
Observed-wind back-trace: where the air at a monitoring station came from, using
only that station's own measured wind (no model).

Why (2026-10-24): on the Oct 23 Hinton-Drinnan H2S/TRS rise the HRDPS wind -
raw and station-corrected - stayed W/NW while Drinnan itself measured SW,
straight from the Mondi mill 2.3 km away, so the LS ranking pointed at the
wrong facility. For sources within a few km of a station, the station's own
wind is the most reliable evidence; this puts it next to every backward run.

  * station: nearest Supabase station within NEAR_KM of the run point
  * wind:    MDS 1-min archive (data/mds_minute, WCAS + ACA) as 10-min vector
             means when the station is there, else the hourly aqhi_data winds
             (hour-ending, treated as the half-hour mid-point)
  * path:    step back in time STEP_MIN at a time, moving upwind; calm
             (< 0.3 m/s) steps don't move. Spatially uniform wind, so it is
             only meaningful for the first few km.
  * nearby sources: NPRI facilities within max(300 m, 25% of their distance
             from the station) of the path, with the time the path passed them.
"""

import datetime as dt
import glob
import math
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd

MINUTE_DIR = "/opt/airquality/data/mds_minute"
NEAR_KM = 3.0
STEP_MIN = 10
CALM_MS = 0.3


def _xy(lat0, lon0, lat, lon):
    return ((lon - lon0) * 111320.0 * math.cos(math.radians(lat0)), (lat - lat0) * 110574.0)


def _ll(lat0, lon0, x, y):
    return lat0 + y / 110574.0, lon0 + x / (111320.0 * math.cos(math.radians(lat0)))


def _minute_winds(name: str, t_a: dt.datetime, t_b: dt.datetime) -> Optional[pd.DataFrame]:
    fn = re.sub(r"[ \u2019']", "_", name) + ".csv.gz"   # archive names: "O\u2019Morrow Station 1" -> O_Morrow_Station_1
    days = sorted({(t_a + dt.timedelta(days=k)).date().isoformat() for k in range((t_b.date() - t_a.date()).days + 1)})
    files = [p for d in days for p in glob.glob(f"{MINUTE_DIR}/*/{d}/{fn}")]
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, usecols=["date", "WS", "WD"], parse_dates=["date"]) for f in files])
    df = df.dropna().drop_duplicates("date").set_index("date").sort_index()
    df = df[(df.index >= t_a - dt.timedelta(minutes=STEP_MIN)) & (df.index <= t_b)]
    if len(df) < 30:
        return None
    ws = df.WS / 3.6
    g = pd.DataFrame({"u": -ws * np.sin(np.radians(df.WD)), "v": -ws * np.cos(np.radians(df.WD))}).resample(f"{STEP_MIN}min").mean().dropna()
    g.index = g.index + pd.Timedelta(minutes=STEP_MIN / 2)   # label at the window mid-point
    return g


def back_trace(obs: List[dict], lat: float, lon: float, t_end: dt.datetime, hours: float,
               sources: List[dict]) -> Optional[dict]:
    """obs: station_obs.fetch_station_winds() rows (hourly fallback + station positions)."""
    near = {}
    for o in obs:
        x, y = _xy(lat, lon, o["lat"], o["lon"])
        d = math.hypot(x, y) / 1000.0
        if d <= NEAR_KM:
            near[o["name"]] = (d, o["lat"], o["lon"])
    if not near:
        return None
    name = min(near, key=lambda n: near[n][0])
    d_st, slat, slon = near[name]
    t_a = t_end - dt.timedelta(hours=hours)
    g = _minute_winds(name, t_a, t_end)
    if g is not None:
        src = f"MDS 1-min, {STEP_MIN}-min vector means"
        tt = np.array([t.timestamp() for t in g.index.to_pydatetime()]); uu, vv = g.u.values, g.v.values
    else:
        rows = sorted((o for o in obs if o["name"] == name), key=lambda o: o["t"])
        if len(rows) < 2:
            return None
        src = "hourly station wind (hour-ending, at mid-hour)"
        tt = np.array([(o["t"] - dt.timedelta(minutes=30)).timestamp() for o in rows])
        uu = np.array([o["u"] for o in rows]); vv = np.array([o["v"] for o in rows])
    # walk back from the run point
    x = y = 0.0
    pts = [(t_end, x, y)]
    t = t_end
    step = dt.timedelta(minutes=STEP_MIN)
    gaps = 0
    while t > t_a:
        tm = (t - step / 2).timestamp()
        k = int(np.argmin(np.abs(tt - tm)))
        if abs(tt[k] - tm) > 3600:
            gaps += 1
            break
        u, v = float(uu[k]), float(vv[k])
        if math.hypot(u, v) >= CALM_MS:
            x -= u * step.total_seconds(); y -= v * step.total_seconds()
        t -= step
        pts.append((t, x, y))
    path = np.array([[p[1], p[2]] for p in pts])
    # sources near the path
    cands = []
    for s in sources:
        sx, sy = _xy(lat, lon, s["lat"], s["lon"])
        dist_st = math.hypot(sx, sy)
        if dist_st > (np.hypot(path[:, 0], path[:, 1]).max() + 1500.0):
            continue
        dd = np.hypot(path[:, 0] - sx, path[:, 1] - sy)
        k = int(np.argmin(dd))
        tol = max(300.0, 0.25 * dist_st)
        if dd[k] <= tol:
            bearing = (math.degrees(math.atan2(sx, sy)) + 360.0) % 360.0
            cands.append({"name": s.get("name"), "company": s.get("company"), "npri_id": s.get("npri_id"), "lat": s["lat"], "lon": s["lon"],
                          "distance_km": round(dist_st / 1000.0, 2), "bearing_deg": round(bearing),
                          "miss_m": round(float(dd[k])), "passed_utc": pts[k][0].isoformat(),
                          "minutes_before": round((t_end - pts[k][0]).total_seconds() / 60)})
    cands.sort(key=lambda c: (c["miss_m"] / max(c["distance_km"] * 1000.0, 300.0), c["distance_km"]))
    coords = []
    for tp, px, py in pts:
        la, lo = _ll(lat, lon, px, py)
        coords.append([round(lo, 6), round(la, 6)])
    end_la, end_lo = _ll(lat, lon, path[-1, 0], path[-1, 1])
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
         "properties": {"kind": "observed_trace", "station": name, "source": src,
                        "times_utc": [p[0].isoformat() for p in pts]}}] + [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": coords[i]},
         "properties": {"kind": "observed_trace_hour", "time_utc": pts[i][0].isoformat()}}
        for i in range(0, len(pts), max(1, 60 // STEP_MIN))]}
    summary = {"station": name, "station_km_from_point": round(d_st, 2), "wind_source": src,
               "steps": len(pts) - 1, "stopped_early": bool(gaps),
               "end_km": round(float(np.hypot(*path[-1])) / 1000.0, 2),
               "end_bearing_deg": round((math.degrees(math.atan2(path[-1, 0], path[-1, 1])) + 360.0) % 360.0),
               "end_lat": round(end_la, 5), "end_lon": round(end_lo, 5),
               "sources_on_path": cands[:10],
               "note": "Path from the station's own measured wind only (uniform wind, no terrain or model). "
                       "Most reliable for sources within a few km; compare with the LS footprint."}
    return {"geojson": geo, "summary": summary}
