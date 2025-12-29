from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import random

from .config import TrajectoryConfig
from .utils_time import ensure_utc

EARTH_R = 6371000.0  # meters

def _mps_to_dlonlat(u, v, lat_deg, dt_s):
    """
    Convert meters/sec (east=u, north=v) into degrees lon/lat over dt_s.
    """
    lat = math.radians(lat_deg)
    dlat = (v * dt_s) / EARTH_R
    # avoid cos(lat)=0 near poles; Alberta is fine but keep safe
    coslat = max(1e-6, math.cos(lat))
    dlon = (u * dt_s) / (EARTH_R * coslat)
    return math.degrees(dlon), math.degrees(dlat)

def _jitter_uv(u, v, cfg: TrajectoryConfig):
    speed = math.hypot(u, v)
    if speed < 1e-9:
        return u, v

    # jitter speed multiplicatively
    speed_j = speed * (1 + random.gauss(0, cfg.sigma_speed_frac))
    speed_j = max(0.0, min(cfg.max_speed_mps, speed_j))

    # jitter direction
    ang = math.atan2(v, u)  # radians
    ang_j = ang + math.radians(random.gauss(0, cfg.sigma_dir_deg))

    uj = speed_j * math.cos(ang_j)
    vj = speed_j * math.sin(ang_j)
    return uj, vj

@dataclass
class ParticlePath:
    lon: list
    lat: list
    t: list

def back_trajectory_ensemble(lon0, lat0, t0: datetime, wind_field, cfg: TrajectoryConfig):
    t0 = ensure_utc(t0)

    steps = int((cfg.lookback_hours * 3600) / cfg.dt_seconds)
    out = []

    for _ in range(cfg.n_particles):
        lon = lon0
        lat = lat0
        t = t0

        lons = [lon]
        lats = [lat]
        ts   = [t.isoformat()]

        for _k in range(steps):
            u, v = wind_field.uv_at(lon, lat, t)
            u, v = _jitter_uv(u, v, cfg)

            # backward step: subtract motion
            dlon, dlat = _mps_to_dlonlat(u, v, lat, cfg.dt_seconds)
            lon -= dlon
            lat -= dlat
            t   -= timedelta(seconds=cfg.dt_seconds)

            lons.append(lon)
            lats.append(lat)
            ts.append(t.isoformat())

        out.append(ParticlePath(lon=lons, lat=lats, t=ts))

    return out
