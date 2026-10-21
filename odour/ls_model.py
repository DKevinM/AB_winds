"""
Lagrangian stochastic (LS) dispersion model - forward and backward.

Replaces the heuristic random walk in backtraj_core.run_back_trajectories
(flat sigma_h/sigma_w, AR(1) persistence 0.5, shear/terrain fudge factors)
with the physically-based model Kevin built as STLS (data/_model2,
Lagrangian Model_b.xlsm "STLModel" macro) - corrected, vectorized, and run
on HRDPS met instead of a single spreadsheet wind:

  * Thomson (1987) well-mixed Langevin equation for vertical velocity,
    Gaussian inhomogeneous turbulence (Flesch & Wilson 1995 simplified
    form for the horizontal components):
        dw = [-w/T_Lw + 1/2 d(sigma_w^2)/dz (1 + w^2/sigma_w^2)] dt
             + sqrt(C0 eps dt) xi
        du' = -u'/T_Lu dt + sqrt(C0 eps dt) xi     (along-wind)
        dv' = -v'/T_Lv dt + sqrt(C0 eps dt) xi     (cross-wind)
  * MOST turbulence (micromet.turbulence) with u*, L, w*, zi derived from
    the HRDPS 10 m wind and 2/40 m temperatures at every particle.
  * Backward mode (Flesch, Wilson & Yee 1995, the WindTrax method): time
    runs backward and the mean wind is reversed. With no mean vertical
    wind and no skewness the vertical process is statistically identical
    in both time directions, so the same vertical SDE is used.
  * Touchdown catalogue (backward mode): every ground reflection records
    position and |w0|. For a surface area source of flux Q (kg/m2/s) the
    receptor concentration is C = Q * (1/N) * sum(2/|w0|) over touchdowns
    inside it - the WindTrax bLS estimator.
  * Perfect reflection at the ground (z = z_r) and at the boundary-layer
    top zi (GRAL does the same), weak residual turbulence above zi.
  * Forward mode only: gravitational settling (Stokes + Cunningham, from
    Kevin's "Particle Settling.doc") and partial ground reflection with
    deposition velocity vd (Wilson, Legg & Thomson 1989 style
    R = (1 - f)/(1 + f), f = vd sqrt(pi/2) / sigma_w).

Fixes relative to the STLS VBA (kept here so the differences are on record):
  - mean wind was added twice (x += wind_vel*dt + (U + ux)*dt);
  - velocities were multiplied by a particle drag coefficient CDrag
    (24/Re etc.), which isn't part of a fluid-particle Langevin model;
  - the horizontal noise used sigma_w for all three components and a
    sigma_w^2/sigma_u^2 factor in the damping term;
  - no d(sigma_w^2)/dz drift term (violates the well-mixed condition in
    the inhomogeneous surface layer - particles pile up near the ground);
  - random_gauss() called Randomize on every draw (timer reseed ->
    correlated "random" numbers).
"""

import math
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import micromet as mm

EARTH_R = 6371000.0
HRDPS_LEVELS = (10.0, 40.0, 80.0, 120.0)


# ---------------------------------------------------------------------------
# Met providers
# ---------------------------------------------------------------------------

class HomogeneousMet:
    """Horizontally homogeneous, steady surface layer - for validation
    against field experiments (Project Prairie Grass) and quick what-ifs.
    Mean wind follows the MOST log profile scaled to U_ref at z_ref, blowing
    TOWARD direction wind_to_deg (0 = north, 90 = east)."""

    def __init__(self, ustar, L, zi=1000.0, z0=0.1, wind_to_deg=90.0,
                 U_ref=None, z_ref=10.0, wstar=None):
        self.ustar = float(ustar)
        self.L = float(L)
        self.zi = float(zi)
        self.z0 = float(z0)
        self.wind_to = math.radians(wind_to_deg)
        if U_ref is None:
            U_ref = self.ustar / mm.KAPPA * (math.log(z_ref / z0) - float(mm.psi_m(z_ref / L)))
        self.U_ref = float(U_ref)
        self.z_ref = float(z_ref)
        if wstar is None:
            wstar = 0.0
            if L < 0:
                # w* from u*, L, zi:  w*^3 = -u*^3 zi / (kappa L)
                wstar = (-self.ustar ** 3 * self.zi / (mm.KAPPA * L)) ** (1.0 / 3.0)
        self.wstar = float(wstar)

    def sample(self, t, lat, lon):
        n = np.shape(lat)
        full = lambda v: np.full(n, v, dtype=float)
        return {"ustar": full(self.ustar), "L": full(self.L), "zi": full(self.zi),
                "wstar": full(self.wstar), "z0": full(self.z0),
                "levels": None, "dir_to": full(self.wind_to), "U_ref": full(self.U_ref),
                "z_ref": self.z_ref}

    def mean_wind(self, s, z):
        ratio = mm.wind_profile_ratio(z, s["z_ref"], s["z0"], s["L"])
        U = s["U_ref"] * ratio
        return U * np.sin(s["dir_to"]), U * np.cos(s["dir_to"])


class HRDPSMet:
    """Vectorized sampler over a backtraj_core.MetStoreV2 (HRDPS JSON
    snapshots). Winds are interpolated in height on the AGL levels
    (10/40/80/120 m) - unlike MetStoreV2.sample(), which was being called
    with height above SEA level and so always returned the 120 m wind in
    Alberta. Below 10 m the MOST log profile is used; above 120 m the
    120 m wind is held."""

    def __init__(self, metstore, z0=0.1, z0_fn=None, correction=None):
        """z0: default roughness (m). z0_fn(lat, lon) -> z0 array overrides it
        per position (landcover.RoughnessMap). correction(t, lat, lon, levels)
        -> levels adjusts the wind levels (station_obs.StationCorrection)."""
        self.met = metstore
        self.z0 = float(z0)
        self.z0_fn = z0_fn
        self.correction = correction
        self.grid = metstore.grid

    def _bilinear(self, field, lat, lon):
        # HRDPS is on a rotated grid stored on a regular lat/lon grid; ~25% of
        # cells (the corners outside the model domain) are NaN, so a result
        # can be NaN there. NaN positions are parked at index 0 and come back
        # NaN too; run_ls drops particles whose met isn't finite.
        g = self.grid
        bad = ~(np.isfinite(lat) & np.isfinite(lon))
        lat = np.where(bad, g.la1, lat)
        lon = np.where(bad, g.lo1, lon)
        i = np.clip((lon - g.lo1) / g.dx, 0.0, g.nx - 1.000001)
        j = np.clip((g.la1 - lat) / g.dy, 0.0, g.ny - 1.000001)
        i0 = np.floor(i).astype(int)
        j0 = np.floor(j).astype(int)
        i1 = np.minimum(i0 + 1, g.nx - 1)
        j1 = np.minimum(j0 + 1, g.ny - 1)
        fi = i - i0
        fj = j - j0
        v0 = field[j0, i0] * (1 - fi) + field[j0, i1] * fi
        v1 = field[j1, i0] * (1 - fi) + field[j1, i1] * fi
        return np.where(bad, np.nan, v0 * (1 - fj) + v1 * fj)

    def _field(self, s0, s1, a, key, lat, lon):
        if key not in s0.fields or key not in s1.fields:
            return None
        return (self._bilinear(s0.fields[key], lat, lon) * (1 - a)
                + self._bilinear(s1.fields[key], lat, lon) * a)

    def sample(self, t, lat, lon):
        s0, s1, a = self.met._bracket(t)
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        lev = {}
        for h in HRDPS_LEVELS:
            u = self._field(s0, s1, a, f"u{int(h)}", lat, lon)
            v = self._field(s0, s1, a, f"v{int(h)}", lat, lon)
            if u is None or v is None:
                raise KeyError(f"HRDPS snapshot missing u/v at {int(h)} m")
            lev[h] = (u, v)
        if self.correction is not None:
            lev = self.correction(t, lat, lon, lev)
        T2 = self._field(s0, s1, a, "tmp2", lat, lon)
        T40 = self._field(s0, s1, a, "tmp40", lat, lon)
        zi = self._field(s0, s1, a, "hpbl", lat, lon)
        if zi is None:
            zi = np.full(lat.shape, 800.0)
        zi = np.where(np.isfinite(zi), zi, 800.0)
        U10 = np.hypot(*lev[10.0])
        if T2 is None or T40 is None:
            # pre-TMP snapshots: assume neutral
            T2 = np.full(lat.shape, 288.0)
            T40 = T2 - 0.0098 * 38.0
        z0 = self.z0_fn(lat, lon) if self.z0_fn is not None else self.z0
        sl = mm.surface_layer_from_profile(U10, T2, T40, z0=z0, zi=zi)
        return {"ustar": sl["ustar"], "L": sl["L"], "zi": sl["zi"],
                "wstar": sl["wstar"], "z0": sl["z0"], "levels": lev}

    def mean_wind(self, s, z):
        lev = s["levels"]
        z = np.asarray(z, dtype=float)
        u = np.empty_like(z)
        v = np.empty_like(z)
        # below 10 m: 10 m wind scaled by the MOST log profile
        low = z < 10.0
        if np.any(low):
            r = mm.wind_profile_ratio(z[low], 10.0, s["z0"][low], s["L"][low])
            u[low] = lev[10.0][0][low] * r
            v[low] = lev[10.0][1][low] * r
        hi = z >= 120.0
        u[hi] = lev[120.0][0][hi]
        v[hi] = lev[120.0][1][hi]
        for lo_h, hi_h in zip(HRDPS_LEVELS[:-1], HRDPS_LEVELS[1:]):
            m = (z >= lo_h) & (z < hi_h)
            if np.any(m):
                f = (z[m] - lo_h) / (hi_h - lo_h)
                u[m] = lev[lo_h][0][m] * (1 - f) + lev[hi_h][0][m] * f
                v[m] = lev[lo_h][1][m] * (1 - f) + lev[hi_h][1][m] * f
        return u, v


def _subset(s, m):
    """Restrict a met-sample dict to particles in boolean mask m."""
    out = {}
    for k, v in s.items():
        if k == "levels":
            out[k] = None if v is None else {h: (uv[0][m], uv[1][m]) for h, uv in v.items()}
        elif isinstance(v, np.ndarray) and v.shape and v.shape[0] == m.shape[0]:
            out[k] = v[m]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class LSConfig:
    C0: float = mm.C0_DEFAULT
    mu: float = 0.1              # turbulence sub-step = mu * T_Lw (Kevin's "mu << 1")
    dt_min: float = 0.5          # s, floor on the sub-step
    dt_adv: float = 60.0         # s, met re-sampling / recording interval
    z_reflect: float = 0.1       # m, ground reflection height (>= z0 is typical)
    reflect_top: bool = True     # reflect at the boundary-layer top
    settling_velocity: float = 0.0   # m/s, forward mode only
    deposition_velocity: float = 0.0  # m/s (excl. settling), forward mode only
    record_every: int = 5        # cloud points every N advection steps
    seed: int = 12345
    # Domain radius (km from the release point). Particles beyond it are
    # dropped: the local-source question (odour, facility screening) is
    # answered inside ~25 km, and regional transport stays with HYSPLIT.
    # None = unlimited (validation runs).
    radius_km: Optional[float] = None
    # Low-wind meander: slow horizontal wind wander the hourly HRDPS field
    # can't resolve (Anfossi et al. 2005; GRAL's 0.3 m/s sigma_v floor).
    # Independent OU velocity per particle, ~0.3 m/s with a 30 min
    # timescale - negligible at 5 m/s, dominant near calm. 0 disables.
    meander_sigma: float = 0.3   # m/s
    meander_T: float = 1800.0    # s
    # The centerline is the mean position of the particles still in play.
    # Once particles start leaving the domain, the mean of the stragglers
    # wanders (survivor bias), so stop the line when fewer than this share
    # remain. The cloud/footprint keep every particle.
    centerline_min_frac: float = 0.5


@dataclass
class LSResult:
    centerlines: List[dict]
    cloud_points: List[tuple]
    touchdowns: List[tuple] = field(default_factory=list)   # (t, lat, lon, |w0|, z0_start)
    n_particles: int = 0
    deposited: int = 0
    crossings: Optional[dict] = None   # validation: x-plane crossings
    exited: int = 0                    # particles dropped at radius_km


def _advance_latlon(lat, lon, dx_e, dy_n):
    dlat = dy_n / EARTH_R * (180.0 / math.pi)
    dlon = dx_e / (EARTH_R * np.maximum(np.cos(np.radians(lat)), 1e-8)) * (180.0 / math.pi)
    return lat + dlat, (lon + dlon + 180.0) % 360.0 - 180.0


def run_ls(met, start_lat, start_lon, start_time_utc, hours=5.0,
           n_particles=500, start_heights_m=(10.0, 40.0, 80.0),
           direction="backward", cfg: Optional[LSConfig] = None,
           crossing_planes_m: Optional[Sequence[float]] = None,
           crossing_dz: float = 0.5, crossing_z: float = 1.5):
    """
    Run the LS model from (start_lat, start_lon) at start_time_utc.

    direction: "backward" (receptor -> where the air came from) or
               "forward" (source -> where it goes).
    crossing_planes_m: validation only - downwind distances (m, along +x
        in HomogeneousMet's frame) at which to tally crosswind-integrated
        concentration Cy/Q at height crossing_z +/- crossing_dz/2.

    Returns LSResult with centerlines/cloud_points in the same shapes
    backtraj_core's GeoJSON writers already consume.
    """
    cfg = cfg or LSConfig()
    rng = np.random.default_rng(cfg.seed)
    sgn = 1.0 if direction == "forward" else -1.0
    n_adv = int(round(hours * 3600.0 / cfg.dt_adv))

    centerlines, cloud_points, touchdowns = [], [], []
    deposited_total = 0
    exited_total = 0
    cos0 = math.cos(math.radians(start_lat))
    r_lim = None if cfg.radius_km is None else cfg.radius_km * 1000.0
    cross = None
    if crossing_planes_m is not None:
        cross = {float(x): 0.0 for x in crossing_planes_m}

    for z_start in start_heights_m:
        n = int(n_particles)
        lat = np.full(n, float(start_lat))
        lon = np.full(n, float(start_lon))
        z = np.full(n, float(z_start))
        xdist = np.zeros(n)                 # along-x distance (validation planes)
        alive = np.ones(n, dtype=bool)

        s = met.sample(start_time_utc, lat, lon)
        turb = mm.turbulence(z, s["ustar"], s["L"], s["zi"], s["wstar"], cfg.C0)
        ua = turb["sigu"] * rng.standard_normal(n)
        uc = turb["sigv"] * rng.standard_normal(n)
        w = turb["sigw"] * rng.standard_normal(n)
        mx = cfg.meander_sigma * rng.standard_normal(n)   # meander, east/north
        my = cfg.meander_sigma * rng.standard_normal(n)

        track = []
        for k in range(n_adv + 1):
            t = start_time_utc + dt.timedelta(seconds=sgn * k * cfg.dt_adv)
            if alive.sum() >= cfg.centerline_min_frac * n:
                track.append((t, float(lat[alive].mean()), float(lon[alive].mean()), float(z[alive].mean())))
            if k % cfg.record_every == 0:
                m_ = int(alive.sum())
                cloud_points.extend(zip([t] * m_, lat[alive].tolist(), lon[alive].tolist(), z[alive].tolist(),
                                        [float(z_start)] * m_))
            if k == n_adv or not np.any(alive):
                break

            # particles that have left the met domain (non-finite wind or
            # stability there) are dropped, like particles leaving radius_km
            s_all = met.sample(t, lat, lon)
            ok = np.isfinite(s_all["ustar"]) & np.isfinite(s_all["L"]) & np.isfinite(s_all["zi"])
            if s_all.get("levels") is not None:
                for uv in s_all["levels"].values():
                    ok &= np.isfinite(uv[0]) & np.isfinite(uv[1])
            left = alive & ~ok
            if np.any(left):
                exited_total += int(left.sum())
                alive &= ok
                if not np.any(alive):
                    break
            idx = np.flatnonzero(alive)
            s = _subset(s_all, alive)
            la, lo, zz = lat[idx], lon[idx], z[idx]
            a_, c_, w_ = ua[idx], uc[idx], w[idx]
            if cfg.meander_sigma > 0:
                rm = math.exp(-cfg.dt_adv / cfg.meander_T)
                qm = cfg.meander_sigma * math.sqrt(1.0 - rm * rm)
                mx[idx] = mx[idx] * rm + qm * rng.standard_normal(idx.size)
                my[idx] = my[idx] * rm + qm * rng.standard_normal(idx.size)
            mxa, mya = mx[idx], my[idx]
            xd = xdist[idx]
            live = np.ones(idx.size, dtype=bool)
            remaining = np.full(idx.size, float(cfg.dt_adv))

            while True:
                act = (remaining > 1e-9) & live
                if not np.any(act):
                    break
                sa = _subset(s, act)
                z_a = zz[act]
                tb = mm.turbulence(z_a, sa["ustar"], sa["L"], sa["zi"], sa["wstar"], cfg.C0)
                ds2 = mm.dsigw2_dz(z_a, sa["ustar"], sa["L"], sa["zi"], sa["wstar"])
                # sub-step: mu*T_Lw, but also small against the sigma_w gradient
                # (dt <= 0.1 / |d sigma_w/dz| = 0.2 sigma_w / |d sigma_w^2/dz|) so the
                # explicit drift term stays stable where sigma_w changes fast
                # (surface and boundary-layer top)
                grad_lim = 0.2 * tb["sigw"] / np.maximum(np.abs(ds2), 1e-9)
                step = np.minimum(remaining[act],
                                  np.maximum(np.minimum(cfg.mu * tb["TLw"], grad_lim), cfg.dt_min))

                # --- velocities: exact OU damping + Thomson drift (vertical)
                for arr, TL, sig in ((a_, tb["TLu"], tb["sigu"]), (c_, tb["TLv"], tb["sigv"])):
                    r = np.exp(-step / TL)
                    arr[act] = arr[act] * r + sig * np.sqrt(1.0 - r * r) * rng.standard_normal(step.size)
                wa = w_[act]
                sw = tb["sigw"]
                wa = np.clip(wa, -6.0 * sw, 6.0 * sw)
                r = np.exp(-step / tb["TLw"])
                drift = 0.5 * ds2 * (1.0 + wa * wa / sw ** 2) * step
                wa = wa * r + drift + sw * np.sqrt(1.0 - r * r) * rng.standard_normal(step.size)
                wa = np.clip(wa, -6.0 * sw, 6.0 * sw)

                # --- mean wind at particle height, rotate fluctuations to E/N
                Ue, Vn = met.mean_wind(sa, z_a)
                spd = np.hypot(Ue, Vn)
                ex = np.where(spd > 1e-6, Ue / np.maximum(spd, 1e-6), 1.0)
                ey = np.where(spd > 1e-6, Vn / np.maximum(spd, 1e-6), 0.0)
                ue = Ue + a_[act] * ex - c_[act] * ey + mxa[act]
                vn = Vn + a_[act] * ey + c_[act] * ex + mya[act]

                dxe = sgn * ue * step
                dyn = sgn * vn * step
                la_new, lo_new = _advance_latlon(la[act], lo[act], dxe, dyn)

                # --- vertical move, settling (forward only)
                vs = cfg.settling_velocity if sgn > 0 else 0.0
                z_new = z_a + wa * step - vs * step

                # --- validation planes (x = +east in HomogeneousMet frame)
                if cross is not None:
                    xd_old = xd[act]
                    xd_new = xd_old + dxe
                    for xp in cross:
                        if sgn > 0:
                            hit = (xd_old < xp) & (xd_new >= xp)
                        else:   # backward validation (2026-10-21): plane xp metres UPWIND of the receptor
                            hit = (xd_old > -xp) & (xd_new <= -xp)
                        if np.any(hit):
                            zf = z_a[hit]
                            ok = np.abs(zf - crossing_z) <= crossing_dz / 2.0
                            # Cy/Q contribution = 1 / (N * u_x * dz)
                            cross[xp] += float(np.sum(1.0 / np.maximum(ue[hit][ok], 0.05)) / crossing_dz)
                    xd[act] = xd_new

                # --- ground reflection / deposition / touchdowns
                zr = cfg.z_reflect
                below = z_new < zr
                if np.any(below):
                    sig_g = tb["sigw"][below]
                    if sgn > 0 and (cfg.deposition_velocity + vs) > 0:
                        f = (cfg.deposition_velocity + vs) * math.sqrt(math.pi / 2.0) / sig_g
                        R = np.clip((1.0 - f) / (1.0 + f), 0.0, 1.0)
                        dep = rng.random(R.size) > R
                        bidx = np.flatnonzero(act)[np.flatnonzero(below)]
                        live[bidx[dep]] = False
                        deposited_total += int(dep.sum())
                    if sgn < 0:
                        tt = t + dt.timedelta(seconds=sgn * float(cfg.dt_adv))
                        m_ = int(below.sum())
                        touchdowns.extend(zip([tt] * m_, la_new[below].tolist(), lo_new[below].tolist(),
                                              np.abs(wa[below]).tolist(), [float(z_start)] * m_))
                    z_new = np.where(below, 2.0 * zr - z_new, z_new)
                    wa = np.where(below, -wa, wa)

                # --- boundary-layer top reflection (only for particles that were below it)
                if cfg.reflect_top:
                    top = (z_new > sa["zi"]) & (z_a <= sa["zi"])
                    if np.any(top):
                        z_new = np.where(top, 2.0 * sa["zi"] - z_new, z_new)
                        wa = np.where(top, -wa, wa)
                z_new = np.maximum(z_new, zr)

                la[act], lo[act], zz[act], w_[act] = la_new, lo_new, z_new, wa
                remaining[act] -= step

            if r_lim is not None:
                dx_ = np.radians(lo - start_lon) * EARTH_R * cos0
                dy_ = np.radians(la - start_lat) * EARTH_R
                out = live & (np.hypot(dx_, dy_) > r_lim)
                exited_total += int(out.sum())
                live &= ~out
            lat[idx], lon[idx], z[idx] = la, lo, zz
            ua[idx], uc[idx], w[idx] = a_, c_, w_
            xdist[idx] = xd
            alive[idx] = live

        centerlines.append({"z0": float(z_start), "track": track})

    if cross is not None:
        n_tot = n_particles * len(start_heights_m)
        cross = {xp: v / n_tot for xp, v in cross.items()}
    return LSResult(centerlines, cloud_points, touchdowns,
                    n_particles=int(n_particles) * len(start_heights_m),
                    deposited=deposited_total, crossings=cross, exited=exited_total)


# ---------------------------------------------------------------------------
# Parallel runs (2026-10-21): split the particles across forked worker
# processes. Each worker inherits `met` through fork (nothing is pickled on
# the way in) and runs run_ls on its share with its own seed; results are
# merged. Statistically the same as one run_ls call with all the particles,
# not bit-identical (different random streams). Linux only (fork).
# ---------------------------------------------------------------------------

_PAR = {}


def _par_worker(i):
    a = _PAR
    cfg = a["cfg"]
    cfg_i = LSConfig(**{**cfg.__dict__, "seed": cfg.seed + 7919 * (i + 1)})
    return run_ls(a["met"], a["lat"], a["lon"], a["t"], hours=a["hours"], n_particles=a["chunks"][i],
                  start_heights_m=a["heights"], direction=a["direction"], cfg=cfg_i, **a["kw"])


def run_ls_parallel(met, start_lat, start_lon, start_time_utc, hours=5.0, n_particles=500,
                    start_heights_m=(10.0, 40.0, 80.0), direction="backward",
                    cfg: Optional[LSConfig] = None, workers: Optional[int] = None, **kw):
    """run_ls with the particles split across `workers` processes (default: cores - 1, max 4).
    Same arguments and LSResult as run_ls. Centrelines are the mean of the workers' centrelines
    at each time (equal particle shares)."""
    import multiprocessing as mp
    import os as _os
    cfg = cfg or LSConfig()
    workers = workers or max(1, min(4, (_os.cpu_count() or 2) - 1))
    workers = min(workers, max(1, int(n_particles) // 50))
    if workers <= 1:
        return run_ls(met, start_lat, start_lon, start_time_utc, hours=hours, n_particles=n_particles,
                      start_heights_m=start_heights_m, direction=direction, cfg=cfg, **kw)
    base, extra = divmod(int(n_particles), workers)
    chunks = [base + (1 if i < extra else 0) for i in range(workers)]
    _PAR.clear()
    _PAR.update(met=met, lat=start_lat, lon=start_lon, t=start_time_utc, hours=hours, chunks=chunks,
                heights=start_heights_m, direction=direction, cfg=cfg, kw=kw)
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_par_worker, range(workers))
    _PAR.clear()

    centerlines = []
    for h in range(len(start_heights_m)):
        by_t = {}
        for p in parts:
            for (t, la, lo, z) in p.centerlines[h]["track"]:
                by_t.setdefault(t, []).append((la, lo, z))
        track = [(t, *(float(np.mean([v[j] for v in vals])) for j in range(3)))
                 for t, vals in sorted(by_t.items(), key=lambda kv: kv[0], reverse=(direction != "forward"))]
        centerlines.append({"z0": float(start_heights_m[h]), "track": track})
    cross = None
    if parts[0].crossings is not None:          # each part is already normalised by its own N
        cross = {xp: float(sum(p.crossings[xp] * c for p, c in zip(parts, chunks)) / sum(chunks))
                 for xp in parts[0].crossings}
    return LSResult(centerlines,
                    [c for p in parts for c in p.cloud_points],
                    [td for p in parts for td in p.touchdowns],
                    n_particles=sum(p.n_particles for p in parts),
                    deposited=sum(p.deposited for p in parts), crossings=cross,
                    exited=sum(p.exited for p in parts))


# ---------------------------------------------------------------------------
# Footprint (WindTrax touchdown estimator) -> GeoJSON
# ---------------------------------------------------------------------------

def touchdown_footprint_geojson(result: LSResult, cell_size_deg: float = 0.01,
                                min_touchdowns: int = 2) -> dict:
    """
    Bin backward-run touchdowns into lat/lon cells. Per cell:

      cq_flux  = (1/N) sum(2/|w0|)      [s/m]   C at the receptor per unit
                 surface flux (kg/m2/s) emitted uniformly from that cell
      cq_point = cq_flux / cell_area    [s/m3]  C/Q at the receptor for a
                 point source of Q (kg/s) at ground level in that cell
      weight   = cell's share of the total footprint (0-1, sums to 1)

    |w0| is floored at 0.1 sigma-ish (0.02 m/s) so a near-zero touchdown
    velocity can't dominate - the known weakness of the 2/|w0| estimator.
    """
    from collections import defaultdict
    acc = defaultdict(float)
    cnt = defaultdict(int)
    N = max(result.n_particles, 1)
    for (_t, lat, lon, w0, _z0) in result.touchdowns:
        key = (math.floor(lon / cell_size_deg), math.floor(lat / cell_size_deg))
        acc[key] += 2.0 / max(w0, 0.02)
        cnt[key] += 1
    total = sum(acc.values()) or 1.0
    feats = []
    for (i, j), s in acc.items():
        if cnt[(i, j)] < min_touchdowns:
            continue
        lon0, lat0 = i * cell_size_deg, j * cell_size_deg
        lon1, lat1 = lon0 + cell_size_deg, lat0 + cell_size_deg
        latc = math.radians(0.5 * (lat0 + lat1))
        area = (EARTH_R * math.radians(cell_size_deg)) ** 2 * math.cos(latc)
        cq_flux = s / N
        feats.append({
            "type": "Feature",
            "properties": {"touchdowns": cnt[(i, j)], "cq_flux_s_per_m": cq_flux,
                           "cq_point_s_per_m3": cq_flux / area, "weight": s / total},
            "geometry": {"type": "Polygon", "coordinates": [[
                [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]]},
        })
    return {"type": "FeatureCollection", "features": feats}
