"""
Gaussian puff model - the time-varying successor to gaussian_plume.py's
hourly steady-state plume (the CALPUFF idea, kept small).

Puffs are released every `release_dt` seconds and carried by the same
HRDPS (optionally station-corrected) wind the LS model uses, sampled at
each puff's own position and height, so a plume can bend, stall or swing
with the wind inside the 25 km domain instead of pointing one way for an
hour.

Puff growth, two options:
  sigma_mode="similarity" (default) - continuous, from the same MOST
      turbulence the LS model uses (micromet.turbulence), via Taylor's
      statistical theory in incremental form:
          d(sigma_y^2)/dt = 2 sigma_v^2 T_Lv (1 - exp(-age/T_Lv)) + meander
          d(sigma_z^2)/dt = 2 sigma_w^2 T_Lw (1 - exp(-age/T_Lw))
      evaluated at the puff's effective height z_e = H + sqrt(2/pi) sigma_z
      (the mean height of a ground-reflected Gaussian), so a ground-level
      puff "sees" growing eddies as it deepens - no jump between classes.
  sigma_mode="briggs" - Briggs rural sigma(x) of travel distance for the
      PG class at the puff (matches gaussian_plume.py; for comparison).

Concentration: each puff's mass m = Q * release_dt spreads as
    C = m / (2 pi sigma_y^2) exp(-r^2 / 2 sigma_y^2) * V(z; H, sigma_z, zi)
with V the ground + mixing-lid image sum from gaussian_plume. Each
advection step is sub-sampled along the puff's path so a fast, narrow
puff paints a continuous trail instead of dots.

direction="backward" releases puffs from a receptor and runs the wind in
reverse; by source-receptor reciprocity the resulting field is C_R/Q_S
for a unit source in each cell (same units as the LS footprint).
"""

import datetime as dt
import math
from typing import Optional

import numpy as np

import micromet as mm
import gaussian_plume as gp

EARTH_R = 6371000.0
ZE_FACTOR = math.sqrt(2.0 / math.pi)   # mean height of a ground-reflected Gaussian / sigma_z


def _met_at(met, t, lat, lon, z):
    """Turbulence + mean wind at (lat, lon, z) for arrays."""
    s = met.sample(t, lat, lon)
    ue, vn = met.mean_wind(s, z)
    return s, ue, vn


def grow_similarity(sy2, sz2, age, step, H, s, C0=mm.C0_DEFAULT,
                    meander_sigma=0.3, meander_T=1800.0):
    """Advance puff variances by one step (arrays). s: met sample dict."""
    z_e = H + ZE_FACTOR * np.sqrt(sz2)
    tb = mm.turbulence(np.maximum(z_e, 1.0), s["ustar"], s["L"], s["zi"], s["wstar"], C0)
    a_mid = age + 0.5 * step
    dy = 2 * tb["sigv"] ** 2 * tb["TLv"] * (1 - np.exp(-a_mid / tb["TLv"]))
    if meander_sigma > 0:
        dy = dy + 2 * meander_sigma ** 2 * meander_T * (1 - np.exp(-a_mid / meander_T))
    dz = 2 * tb["sigw"] ** 2 * tb["TLw"] * (1 - np.exp(-a_mid / tb["TLw"]))
    return sy2 + dy * step, sz2 + dz * step


def similarity_sigma_z(x, U, H, ustar, L, zi=1000.0, wstar=0.0, step=1.0):
    """sigma_z (m) after travel distance x (m) at speed U, integrating
    grow_similarity - used for the steady-plume Prairie Grass check."""
    s = {"ustar": np.array([ustar]), "L": np.array([L]), "zi": np.array([zi]), "wstar": np.array([wstar])}
    sy2 = np.array([0.0])
    sz2 = np.array([0.0])
    t_end = x / max(U, 0.3)
    age = 0.0
    while age < t_end:
        st = min(step * (1 + age / 20.0), t_end - age)   # small steps near the source
        sy2, sz2 = grow_similarity(sy2, sz2, np.array([age]), st, H, s, meander_sigma=0.0)
        age += st
    return float(np.sqrt(sz2[0]))


def run_puffs(met, lat0, lon0, start_time, hours=3.0, direction="forward",
              H=2.0, z_eval=1.5, Q=1.0, release_dt=60.0, step_dt=60.0,
              radius_km=25.0, cell_m=250.0, sigma_mode="similarity",
              max_age_h: Optional[float] = None, urban=False, spinup_h: float = 2.0):
    """
    Release puffs from (lat0, lon0) for `hours`, follow them until they
    leave radius_km (or max_age_h), and return the time-averaged field
    on a grid around the release point:
        forward:  C/Q (s/m3) at height z_eval from a source at height H
        backward: C_R/Q_S (s/m3) for a receptor at z_eval and a unit
                  source at height H in each cell
    spinup_h: puffs start this long before the averaging window so a
    continuous source already has its plume in place at start_time
    (without it the far field reads low by the travel time).
    Returns dict(lat, lon, cq, cell_m, n_puffs, hours).
    """
    sgn = 1.0 if direction == "forward" else -1.0
    R = radius_km * 1000.0
    n = int(2 * R / cell_m) + 1
    half = n // 2
    acc = np.zeros((n, n))
    cos0 = math.cos(math.radians(lat0))
    max_age = (max_age_h if max_age_h is not None else hours + 6.0) * 3600.0

    # puff state (local metres east/north of the release point)
    px = np.zeros(0); py = np.zeros(0); age = np.zeros(0)
    sy2 = np.zeros(0); sz2 = np.zeros(0); trav = np.zeros(0)
    spin = spinup_h * 3600.0
    n_rel = int(round((hours * 3600.0 + spin) / release_dt))
    released = 0
    t_sim = -spin
    total_T = hours * 3600.0
    mass = Q * release_dt

    while True:
        # release puffs due in this step
        while released < n_rel and released * release_dt - spin <= t_sim:
            px = np.append(px, 0.0); py = np.append(py, 0.0); age = np.append(age, 0.0)
            sy2 = np.append(sy2, 0.01); sz2 = np.append(sz2, 0.01); trav = np.append(trav, 0.0)
            released += 1
        if px.size == 0 and released >= n_rel:
            break
        t = start_time + dt.timedelta(seconds=sgn * t_sim)
        lat = lat0 + np.degrees(py / EARTH_R)
        lon = lon0 + np.degrees(px / (EARTH_R * cos0))
        z_adv = np.maximum(H + ZE_FACTOR * np.sqrt(sz2), 2.0)
        s, ue, vn = _met_at(met, t, lat, lon, z_adv)
        ok = np.isfinite(ue) & np.isfinite(vn) & np.isfinite(s["ustar"]) & np.isfinite(s["L"]) & np.isfinite(s["zi"])
        if not ok.all():               # puffs that left the met domain are dropped
            px, py, age, sy2, sz2, trav = px[ok], py[ok], age[ok], sy2[ok], sz2[ok], trav[ok]
            s = {k: (v[ok] if isinstance(v, np.ndarray) and v.shape and v.shape[0] == ok.shape[0] else v)
                 for k, v in s.items() if k != "levels"}
            ue, vn = ue[ok], vn[ok]
            if px.size == 0:
                t_sim += step_dt
                if t_sim >= total_T:
                    break
                continue
        spd = np.maximum(np.hypot(ue, vn), 0.3)

        # grow
        if sigma_mode == "briggs":
            pg = mm.pg_class_from_L(s["L"], s["z0"])
            trav_new = trav + spd * step_dt
            sy_n = np.empty_like(trav_new); sz_n = np.empty_like(trav_new)
            for c in set(pg.tolist()):
                m = pg == c
                a, b = gp.sigmas(trav_new[m], c, urban)
                sy_n[m], sz_n[m] = a, b
            sy2_new, sz2_new = sy_n ** 2, sz_n ** 2
            trav = trav_new
        else:
            sy2_new, sz2_new = grow_similarity(sy2, sz2, age, step_dt, H, s)

        # move + paint the path (sub-sampled so narrow puffs leave a trail)
        dxe = sgn * ue * step_dt
        dyn = sgn * vn * step_dt
        n_sub = np.clip(np.ceil(np.hypot(dxe, dyn) / (0.5 * np.sqrt(sy2_new) + 1e-6)), 1, 10).astype(int)
        for i in range(px.size):
            ns = n_sub[i]
            for k in range(ns):
                f = (k + 0.5) / ns
                cx = px[i] + f * dxe[i]
                cy = py[i] + f * dyn[i]
                syk = math.sqrt(sy2[i] + f * (sy2_new[i] - sy2[i]))
                szk = math.sqrt(sz2[i] + f * (sz2_new[i] - sz2[i]))
                if abs(cx) > R + 4 * syk or abs(cy) > R + 4 * syk:
                    continue
                r = 4 * syk
                i0 = max(int((cx - r) / cell_m) + half, 0); i1 = min(int((cx + r) / cell_m) + half + 1, n)
                j0 = max(int((cy - r) / cell_m) + half, 0); j1 = min(int((cy + r) / cell_m) + half + 1, n)
                if i0 >= i1 or j0 >= j1:
                    continue
                xs = (np.arange(i0, i1) - half) * cell_m - cx
                ys = (np.arange(j0, j1) - half) * cell_m - cy
                g = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (2 * syk ** 2)) / (2 * math.pi * syk ** 2)
                # a puff narrower than a cell would be missed by point sampling -
                # renormalise so the cells hold its full horizontal mass
                gs = g.sum() * cell_m ** 2
                if gs > 0:
                    g = g / max(gs, 1e-12) if syk < cell_m else g
                if not (0.0 <= t_sim < total_T):
                    continue           # spin-up: puffs move but aren't averaged
                V = float(gp._vertical_term(z_eval, H, szk, float(s["zi"][i])))
                acc[j0:j1, i0:i1] += mass * g * V * (step_dt / ns)

        px = px + dxe; py = py + dyn; age = age + step_dt
        sy2, sz2 = sy2_new, sz2_new
        keep = (np.hypot(px, py) <= R) & (age <= max_age)
        px, py, age, sy2, sz2, trav = px[keep], py[keep], age[keep], sy2[keep], sz2[keep], trav[keep]
        t_sim += step_dt
        if t_sim >= total_T:
            break                      # averaging window over

    cq = acc / total_T
    off = (np.arange(n) - half) * cell_m
    X, Y = np.meshgrid(off, off)
    return {"lat": lat0 + np.degrees(Y / EARTH_R),
            "lon": lon0 + np.degrees(X / (EARTH_R * cos0)),
            "cq": cq, "cell_m": cell_m, "n_puffs": released, "hours": hours}
