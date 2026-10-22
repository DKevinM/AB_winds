"""
Observed station winds for correcting HRDPS near the receptor.

Source: the Alberta ambient network feed already ingested into Supabase
aqhi_data ("Wind Speed" in km/h, "Wind Direction" in degrees FROM, hourly)
with coordinates from the stations table. Checked 2026-10-13 against
HRDPS 10 m at 77 stations: speeds are km/h (median obs/HRDPS ratio 3.2),
and HRDPS direction was off by a median 21 deg (75th percentile 35 deg) -
the error this correction targets.

StationCorrection is plugged into ls_model.HRDPSMet(correction=...). For
each station-hour it stores the residual (observed - HRDPS 10 m wind),
then at any particle position blends nearby residuals:

    w_i  = exp(-(d_i / length_km)^2)
    corr = sum(w_i * residual_i) / (sum(w_i) + k0)

k0 fades the correction to zero away from stations (one station on top of
the particle gives ~90% of its residual). The correction is applied in
full at 10-40 m and tapers to 60% at 120 m (stations measure ~10 m; the
boundary layer above still shares much of a direction error).
"""

import datetime as dt
import math
import os
from typing import List, Optional

import numpy as np
import requests

EARTH_R = 6371000.0
KMH_TO_MS = 1.0 / 3.6
LEVEL_WEIGHT = {10.0: 1.0, 40.0: 1.0, 80.0: 0.8, 120.0: 0.6}


def _headers():
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def _dist_m(lat0, lon0, lat, lon):
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R
    return np.hypot(x, y)


def fetch_station_winds(lat: float, lon: float, t_start: dt.datetime, t_end: dt.datetime,
                        radius_km: float = 60.0) -> List[dict]:
    """Hourly station winds within radius_km of (lat, lon) between t_start
    and t_end (UTC). Returns dicts: name, lat, lon, t, u, v (m/s, the
    direction the air moves toward), ws (m/s)."""
    url = os.environ["SUPABASE_URL"]
    st = requests.get(f"{url}/rest/v1/stations?select=StationName,Latitude,Longitude",
                      headers=_headers(), timeout=30).json()
    near = {}
    for r in st:
        try:
            la, lo = float(r["Latitude"]), float(r["Longitude"])
        except (TypeError, ValueError):
            continue
        if float(_dist_m(lat, lon, la, lo)) <= radius_km * 1000.0:
            near[r["StationName"]] = (la, lo)
    if not near:
        return []
    names = ",".join('"' + n.replace('"', '') + '"' for n in near)
    t0 = (t_start - dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
    t1 = (t_end + dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
    params = {
        "select": "StationName,ParameterName,ReadingDate,Value",
        "ParameterName": 'in.("Wind Speed","Wind Direction")',
        "StationName": f"in.({names})",
        "and": f"(ReadingDate.gte.{t0},ReadingDate.lte.{t1})",
        "limit": "10000",
    }
    rows = requests.get(f"{url}/rest/v1/aqhi_data", params=params, headers=_headers(), timeout=60).json()
    if isinstance(rows, dict):          # PostgREST error object
        raise RuntimeError(f"aqhi_data query failed: {rows.get('message')}")
    pair = {}
    for r in rows:
        pair.setdefault((r["StationName"], r["ReadingDate"]), {})[r["ParameterName"]] = r["Value"]
    out = []
    for (name, ts), v in pair.items():
        ws, wd = v.get("Wind Speed"), v.get("Wind Direction")
        if ws is None or wd is None:
            continue
        ws = float(ws) * KMH_TO_MS
        wd = float(wd)
        if not (0.0 <= ws <= 40.0) or not (0.0 <= wd <= 360.0):
            continue
        if ws < 0.3:                    # calm: direction meaningless, speed ~0
            u = v_ = 0.0
        else:
            u = -ws * math.sin(math.radians(wd))
            v_ = -ws * math.cos(math.radians(wd))
        t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        la, lo = near[name]
        out.append({"name": name, "lat": la, "lon": lo, "t": t, "u": u, "v": v_, "ws": ws})
    return out


class StationCorrection:
    """Callable for HRDPSMet(correction=...): (t, lat, lon, levels) -> levels."""

    def __init__(self, base_met, obs: List[dict], length_km: float = 15.0, k0: float = 0.1,
                 max_gap_h: float = 1.5, exposure: bool = False):
        self.length = length_km * 1000.0
        self.k0 = k0
        self.max_gap = max_gap_h * 3600.0
        self.n_obs = len(obs)
        if not obs:
            self.lat = np.zeros(0)
            return
        lat = np.array([o["lat"] for o in obs])
        lon = np.array([o["lon"] for o in obs])
        tt = [o["t"] for o in obs]
        du = np.zeros(len(obs))
        dv = np.zeros(len(obs))
        so = np.zeros(len(obs))     # observed speed
        sm = np.zeros(len(obs))     # model 10 m speed at the station
        for i, o in enumerate(obs):
            s = base_met.sample(o["t"], lat[i:i + 1], lon[i:i + 1])
            u10, v10 = s["levels"][10.0]
            du[i] = o["u"] - float(u10[0])
            dv[i] = o["v"] - float(v10[0])
            so[i] = math.hypot(o["u"], o["v"])
            sm[i] = math.hypot(float(u10[0]), float(v10[0]))
        self.lat, self.lon, self.du, self.dv = lat, lon, du, dv
        self.epoch = np.array([t.timestamp() for t in tt])
        self.names = [o["name"] for o in obs]
        self.stations = sorted(set(self.names))
        # Exposure weight (2026-10-22): a sheltered station (median observed speed well below the model's at
        # that spot over this run) mostly measures its own shelter - e.g. Hinton-Drinnan, mean 3.7 km/h - and
        # pulling the field toward it made back-trajectories worse in the known-source test (Drinnan 74% -> 61%
        # within 22.5 deg of the mill). ratio >= 0.7 full weight, <= 0.3 excluded, linear between; < 3 hours = 1.
        # OFF by default: in the known-source test it did not help Drinnan (61% either way) and made ECCO worse
        # (23% -> 15% within 22.5 deg), 2026-10-22. Kept for experiments: StationCorrection(..., exposure=True).
        self.exposure = {}
        for n in self.stations:
            m = np.array([x == n for x in self.names])
            r = float(np.median(so[m]) / max(np.median(sm[m]), 0.5)) if m.sum() >= 3 else 1.0
            self.exposure[n] = (float(np.clip((r - 0.3) / 0.4, 0.0, 1.0)) if exposure else 1.0, round(r, 2))
        self.wexp = np.array([self.exposure[n][0] for n in self.names])

    def residual_summary(self):
        """Per-station mean residual (m/s) - diagnostics for model_info."""
        out = []
        for n in self.stations:
            m = np.array([x == n for x in self.names])
            out.append({"station": n, "lat": float(self.lat[m][0]), "lon": float(self.lon[m][0]),
                        "hours": int(m.sum()), "mean_du": float(self.du[m].mean()),
                        "mean_dv": float(self.dv[m].mean()),
                        "speed_ratio": self.exposure[n][1], "exposure_weight": round(self.exposure[n][0], 2)})
        return out

    def correction_at(self, t: dt.datetime, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        if self.lat.size == 0:
            return np.zeros(lat.shape), np.zeros(lat.shape)
        gap = np.abs(self.epoch - t.timestamp())
        # per station keep only the nearest hour, and only if within max_gap
        use = gap <= self.max_gap
        best = {}
        for i in np.flatnonzero(use):
            n = self.names[i]
            if n not in best or gap[i] < gap[best[n]]:
                best[n] = i
        if not best:
            return np.zeros(lat.shape), np.zeros(lat.shape)
        ii = np.array(list(best.values()))
        slat, slon = self.lat[ii], self.lon[ii]
        cos0 = math.cos(math.radians(float(np.mean(slat))))
        dx = np.radians(lon[:, None] - slon[None, :]) * EARTH_R * cos0
        dy = np.radians(lat[:, None] - slat[None, :]) * EARTH_R
        w = np.exp(-(dx ** 2 + dy ** 2) / self.length ** 2) * self.wexp[ii][None, :]
        den = w.sum(axis=1) + self.k0
        return (w @ self.du[ii]) / den, (w @ self.dv[ii]) / den

    def __call__(self, t, lat, lon, levels):
        cu, cv = self.correction_at(t, lat, lon)
        return {h: (uv[0] + LEVEL_WEIGHT.get(h, 0.6) * cu, uv[1] + LEVEL_WEIGHT.get(h, 0.6) * cv)
                for h, uv in levels.items()}
