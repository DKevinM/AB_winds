"""
Steady-state Gaussian plume model - the screening counterpart to the LS
model (ls_model.py). Built from Kevin's gaussian.R / Gaussian1.xls
(data/_model2) and the "Computer Practical - Gaussian Plume Model" notes,
with these changes:

  * sigma_y(x), sigma_z(x) are functions of downwind distance (Briggs 1973
    open-country and urban fits) - gaussian.R evaluated them at a fixed
    x = 10 km only;
  * cross-wind term uses exp(-y^2 / (2 sigma_y^2)) - gaussian.R had
    2*sigma_y (harmless there only because y = 0);
  * mixing-lid reflection (image sources) plus the well-mixed limit
    C = Q / (sqrt(2 pi) sigma_y u zi) once sigma_z > 1.6 zi;
  * Briggs (1975) plume rise for buoyant point sources (optional);
  * settling-plume tilt for particles: H_eff(x) = H - v_s x / u;
  * stability class from the same MOST u*/L the LS model uses
    (micromet.pg_class_from_L), so both models see one consistent met.

Three ways to use it on HRDPS met:
  plume_grid()        source -> gridded C/Q around it (forward)
  receptor_footprint() receptor -> C/Q at the receptor per unit source in
                       each upwind cell (the Gaussian analogue of the LS
                       touchdown footprint - same units, comparable)
  rank_sources()      receptor + list of known sources (e.g. NPRI
                       facilities) -> modelled C/Q from each, ranked
"""

import math
import datetime as dt
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

import micromet as mm

EARTH_R = 6371000.0

# Briggs (1973) sigma fits, x in metres (as in gaussian.R / Hanna et al. 1982)
_BRIGGS_RURAL = {
    "A": (lambda x: 0.22 * x * (1 + 0.0001 * x) ** -0.5, lambda x: 0.20 * x),
    "B": (lambda x: 0.16 * x * (1 + 0.0001 * x) ** -0.5, lambda x: 0.12 * x),
    "C": (lambda x: 0.11 * x * (1 + 0.0001 * x) ** -0.5, lambda x: 0.08 * x * (1 + 0.0002 * x) ** -0.5),
    "D": (lambda x: 0.08 * x * (1 + 0.0001 * x) ** -0.5, lambda x: 0.06 * x * (1 + 0.0015 * x) ** -0.5),
    "E": (lambda x: 0.06 * x * (1 + 0.0001 * x) ** -0.5, lambda x: 0.03 * x * (1 + 0.0003 * x) ** -1),
    "F": (lambda x: 0.04 * x * (1 + 0.0001 * x) ** -0.5, lambda x: 0.016 * x * (1 + 0.0003 * x) ** -1),
}
_BRIGGS_URBAN = {
    "A": (lambda x: 0.32 * x * (1 + 0.0004 * x) ** -0.5, lambda x: 0.24 * x * (1 + 0.001 * x) ** 0.5),
    "B": (lambda x: 0.32 * x * (1 + 0.0004 * x) ** -0.5, lambda x: 0.24 * x * (1 + 0.001 * x) ** 0.5),
    "C": (lambda x: 0.22 * x * (1 + 0.0004 * x) ** -0.5, lambda x: 0.20 * x),
    "D": (lambda x: 0.16 * x * (1 + 0.0004 * x) ** -0.5, lambda x: 0.14 * x * (1 + 0.0003 * x) ** -0.5),
    "E": (lambda x: 0.11 * x * (1 + 0.0004 * x) ** -0.5, lambda x: 0.08 * x * (1 + 0.0015 * x) ** -0.5),
    "F": (lambda x: 0.11 * x * (1 + 0.0004 * x) ** -0.5, lambda x: 0.08 * x * (1 + 0.0015 * x) ** -0.5),
}


def sigmas(x, pg, urban=False):
    """sigma_y, sigma_z (m) at downwind distance x (m) for PG class pg."""
    x = np.maximum(np.asarray(x, dtype=float), 1.0)
    fy, fz = (_BRIGGS_URBAN if urban else _BRIGGS_RURAL)[pg]
    return fy(x), fz(x)


def _vertical_term(z, H, sz, zi, n_images=3):
    """Sum of Gaussian images for ground + lid reflection, /(sqrt(2pi) sz).
    Switches to the well-mixed 1/zi limit once sz > 1.6 zi."""
    z = np.asarray(z, dtype=float)
    H = np.asarray(H, dtype=float)
    sz = np.asarray(sz, dtype=float)
    out = np.zeros(np.broadcast(z, H, sz).shape)
    for nimg in range(-n_images, n_images + 1):
        out = out + np.exp(-(z - H + 2 * nimg * zi) ** 2 / (2 * sz ** 2)) \
                  + np.exp(-(z + H + 2 * nimg * zi) ** 2 / (2 * sz ** 2))
    out = out / (math.sqrt(2 * math.pi) * sz)
    if zi is not None and np.isfinite(zi):
        out = np.where(sz > 1.6 * zi, 1.0 / zi, out)
    return out


def briggs_plume_rise(x, u, F_b, pg, dtheta_dz=None):
    """Briggs (1975) gradual/final rise (m) for buoyancy flux F_b (m4/s3).
    Neutral/unstable: dh = 1.6 F^(1/3) x^(2/3) / u, capped at x_f.
    Stable (E/F): final dh = 2.6 (F / (u s))^(1/3), s = g/T dtheta/dz."""
    x = np.asarray(x, dtype=float)
    u = max(float(u), 0.5)
    if F_b <= 0:
        return np.zeros_like(x)
    if pg in ("E", "F"):
        s = 9.81 / 288.0 * (dtheta_dz if dtheta_dz is not None else (0.02 if pg == "E" else 0.035))
        final = 2.6 * (F_b / (u * s)) ** (1.0 / 3.0)
        grad = 1.6 * F_b ** (1.0 / 3.0) * x ** (2.0 / 3.0) / u
        return np.minimum(grad, final)
    xf = 49.0 * F_b ** 0.625 if F_b < 55 else 119.0 * F_b ** 0.4
    return 1.6 * F_b ** (1.0 / 3.0) * np.minimum(x, xf) ** (2.0 / 3.0) / u


def concentration(x, y, z, Q, u, H, pg, zi=1000.0, urban=False,
                  settling_velocity=0.0, F_b=0.0):
    """Gaussian plume concentration (Q units / m3; with Q in g/s -> g/m3).
    x downwind, y crosswind, z height (m). Zero upwind (x <= 0)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = max(float(u), 0.5)
    xp = np.maximum(x, 1.0)
    sy, sz = sigmas(xp, pg, urban)
    He = H + briggs_plume_rise(xp, u, F_b, pg) - settling_velocity * xp / u
    He = np.maximum(He, 0.0)
    # Q/(2 pi u sy sz) exp(-y^2/2sy^2) [images]  ==  Q/(sqrt(2pi) u sy) exp(..) * V(z)
    c = Q / (math.sqrt(2 * math.pi) * u * sy) * np.exp(-y ** 2 / (2 * sy ** 2)) \
        * _vertical_term(z, He, sz, zi)
    return np.where(x > 0, c, 0.0)


def crosswind_integrated(x, z, H, u, pg, zi=1000.0, urban=False):
    """Cy/Q (s/m2) - integral of C/Q over y. Used for Prairie Grass."""
    u = max(float(u), 0.5)
    _, sz = sigmas(x, pg, urban)
    return float(_vertical_term(z, H, sz, zi) / u)


# ---------------------------------------------------------------------------
# HRDPS-driven helpers
# ---------------------------------------------------------------------------

def _local_xy(lat0, lon0, lat, lon):
    """East/north offsets (m) of (lat, lon) from (lat0, lon0)."""
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R
    return x, y


def hourly_met(met, lat, lon, times: Sequence[dt.datetime], height_m=10.0):
    """Wind speed/direction-of-motion at height_m, PG class, L, zi at a
    point for each time. met is an ls_model.HRDPSMet (or HomogeneousMet)."""
    out = []
    la = np.array([lat], dtype=float)
    lo = np.array([lon], dtype=float)
    for t in times:
        s = met.sample(t, la, lo)
        ue, vn = met.mean_wind(s, np.array([max(height_m, 2.0)]))
        U = float(np.hypot(ue[0], vn[0]))
        out.append({"t": t, "U": U, "dir_to": math.atan2(float(ue[0]), float(vn[0])),
                    "L": float(s["L"][0]), "zi": float(s["zi"][0]),
                    "pg": str(mm.pg_class_from_L(s["L"][0], s["z0"][0])[0]),
                    "ustar": float(s["ustar"][0])})
    return out


def _rotate(dx, dy, dir_to):
    """(east, north) offsets -> (downwind, crosswind) for wind blowing
    toward dir_to (radians from north)."""
    s, c = math.sin(dir_to), math.cos(dir_to)
    return dx * s + dy * c, -dx * c + dy * s


def plume_grid(met, src_lat, src_lon, times, H=10.0, z_rcp=1.5,
               half_width_km=20.0, cell_m=500.0, urban=False, F_b=0.0,
               settling_velocity=0.0):
    """Hour-averaged C/Q (s/m3) on a grid around a source, using the
    HRDPS wind/stability at the source each hour (steady-state per hour,
    the standard screening assumption)."""
    n = int(2 * half_width_km * 1000 / cell_m) + 1
    off = (np.arange(n) - n // 2) * cell_m
    X, Y = np.meshgrid(off, off)
    acc = np.zeros_like(X)
    hm = hourly_met(met, src_lat, src_lon, times, height_m=max(H, 10.0))
    for h in hm:
        xd, yc = _rotate(X, Y, h["dir_to"])
        acc += concentration(xd, yc, z_rcp, 1.0, h["U"], H, h["pg"], h["zi"], urban,
                             settling_velocity, F_b)
    acc /= max(len(hm), 1)
    lat_c = src_lat + np.degrees(Y / EARTH_R)
    lon_c = src_lon + np.degrees(X / (EARTH_R * math.cos(math.radians(src_lat))))
    return {"lat": lat_c, "lon": lon_c, "cq": acc, "cell_m": cell_m, "hours": hm}


def receptor_footprint(met, rcp_lat, rcp_lon, times, src_height=2.0, z_rcp=1.5,
                       half_width_km=30.0, cell_m=500.0, urban=False):
    """Gaussian back-footprint: for every cell S around the receptor R,
    C_R/Q_S (s/m3) from a unit point source at height src_height in S,
    averaged over `times`, using the HRDPS wind/stability at R each hour.
    Same units as ls_model.touchdown_footprint_geojson's cq_point, so the
    two maps are directly comparable."""
    n = int(2 * half_width_km * 1000 / cell_m) + 1
    off = (np.arange(n) - n // 2) * cell_m
    X, Y = np.meshgrid(off, off)        # S relative to R
    acc = np.zeros_like(X)
    hm = hourly_met(met, rcp_lat, rcp_lon, times, height_m=10.0)
    for h in hm:
        # vector S -> R is (-X, -Y): downwind distance from source to receptor
        xd, yc = _rotate(-X, -Y, h["dir_to"])
        acc += concentration(xd, yc, z_rcp, 1.0, h["U"], src_height, h["pg"], h["zi"], urban)
    acc /= max(len(hm), 1)
    lat_c = rcp_lat + np.degrees(Y / EARTH_R)
    lon_c = rcp_lon + np.degrees(X / (EARTH_R * math.cos(math.radians(rcp_lat))))
    return {"lat": lat_c, "lon": lon_c, "cq": acc, "cell_m": cell_m, "hours": hm}


def rank_sources(met, rcp_lat, rcp_lon, times, sources: Iterable[dict], z_rcp=1.5,
                 urban=False) -> List[dict]:
    """Modelled C/Q at the receptor from each known source, averaged over
    `times`. sources: dicts with lat, lon and optional name, height_m,
    Q (g/s) and F_b (buoyancy flux). Returns them sorted, highest first,
    with cq_s_per_m3 (and c_ug_m3 when Q is given)."""
    hm = hourly_met(met, rcp_lat, rcp_lon, times, height_m=10.0)
    out = []
    for src in sources:
        dx, dy = _local_xy(src["lat"], src["lon"], rcp_lat, rcp_lon)   # S -> R
        H = float(src.get("height_m", 10.0))
        cq = 0.0
        for h in hm:
            xd, yc = _rotate(float(dx), float(dy), h["dir_to"])
            cq += float(concentration(xd, yc, z_rcp, 1.0, h["U"], H, h["pg"], h["zi"],
                                      urban, 0.0, float(src.get("F_b", 0.0))))
        cq /= max(len(hm), 1)
        row = dict(src)
        row["cq_s_per_m3"] = cq
        row["distance_km"] = float(math.hypot(dx, dy) / 1000.0)
        if "Q" in src:
            row["c_ug_m3"] = cq * float(src["Q"]) * 1e6
        out.append(row)
    out.sort(key=lambda r: r["cq_s_per_m3"], reverse=True)
    return out


def grid_to_geojson(g: dict, min_frac: float = 1e-3) -> dict:
    """Grid from plume_grid / receptor_footprint -> GeoJSON polygons
    (cells below min_frac of the max are dropped)."""
    cq = g["cq"]
    vmax = float(cq.max()) if cq.size else 0.0
    feats = []
    if vmax <= 0:
        return {"type": "FeatureCollection", "features": feats}
    lat, lon = g["lat"], g["lon"]
    dlat = abs(lat[1, 0] - lat[0, 0]) / 2 if lat.shape[0] > 1 else 0.0025
    dlon = abs(lon[0, 1] - lon[0, 0]) / 2 if lon.shape[1] > 1 else 0.0025
    total = float(cq.sum())
    for j, i in zip(*np.nonzero(cq >= min_frac * vmax)):
        la, lo = float(lat[j, i]), float(lon[j, i])
        feats.append({"type": "Feature",
                      "properties": {"cq_s_per_m3": float(cq[j, i]), "weight": float(cq[j, i]) / total},
                      "geometry": {"type": "Polygon", "coordinates": [[
                          [lo - dlon, la - dlat], [lo + dlon, la - dlat], [lo + dlon, la + dlat],
                          [lo - dlon, la + dlat], [lo - dlon, la - dlat]]]}})
    return {"type": "FeatureCollection", "features": feats}
