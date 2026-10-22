"""
Uncertainty ensemble for the LS model (2026-10-21).

One LS run answers "where did the air come from, if the wind is exactly right". The wind isn't:
measured HRDPS direction errors range from a median 6.5 deg (Edmonton, station-corrected) to only 74%
within 45 deg (Hinton-Drinnan, raw, sheltered valley). This module reruns the model with the mean wind
rotated and scaled per member, so a footprint becomes a probability:

    "the air passed over this area in X of N plausible wind fields".

Each member: direction offset ~ N(0, sigma_dir_deg), speed factor ~ lognormal(0, sigma_speed), held
constant through the run (model wind errors persist for hours), plus its own random seed. Defaults
sigma_dir_deg = 20 (about 97% of members within 45 deg) and sigma_speed = 0.3 sit between a dense
station network and a sheltered valley; pass smaller values where stations are dense.

Members run in parallel (fork). Turbulence (u*, L, zi) is not perturbed.
"""
import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import ls_model as ls

DEG_M = 111320.0


class PerturbedMet:
    """Wraps an HRDPSMet/HomogeneousMet: same turbulence, mean wind rotated by dtheta (deg,
    clockwise) and scaled by fspeed."""

    def __init__(self, base, dtheta_deg, fspeed):
        self.base = base
        self.dtheta = math.radians(dtheta_deg)
        self.fspeed = fspeed

    def sample(self, t, lat, lon):
        return self.base.sample(t, lat, lon)

    def mean_wind(self, s, z):
        ue, vn = self.base.mean_wind(s, z)
        c, sn = math.cos(self.dtheta), math.sin(self.dtheta)
        # clockwise rotation of the (east, north) vector, then scale
        return self.fspeed * (ue * c + vn * sn), self.fspeed * (-ue * sn + vn * c)

    def __getattr__(self, name):          # anything else (correction, z0_fn, ...) from the base
        return getattr(self.base, name)


@dataclass
class Member:
    dtheta_deg: float
    fspeed: float
    grid: np.ndarray          # touchdown weight per cell, normalised to sum 1
    n_touchdowns: int


@dataclass
class EnsembleResult:
    members: List[Member]
    lat0: float               # grid centre (the receptor)
    lon0: float
    cell_m: float
    half_n: int               # grid is (2*half_n+1)^2 cells
    sigma_dir_deg: float
    sigma_speed: float

    def _ij(self, lat, lon):
        x = (lon - self.lon0) * DEG_M * math.cos(math.radians(self.lat0))
        y = (lat - self.lat0) * DEG_M
        return int(round(y / self.cell_m)) + self.half_n, int(round(x / self.cell_m)) + self.half_n

    def mean_footprint(self):
        return np.mean([m.grid for m in self.members], axis=0)

    def core_masks(self, mass=0.5):
        """Per member: cells making up its smallest region holding `mass` of the footprint."""
        out = []
        for m in self.members:
            g = m.grid.ravel()
            order = np.argsort(g)[::-1]
            keep = np.zeros(g.size, bool)
            if g.sum() > 0:
                c = np.cumsum(g[order])
                keep[order[: int(np.searchsorted(c, mass)) + 1]] = True
            out.append(keep.reshape(m.grid.shape))
        return np.array(out)

    def agreement(self, mass=0.5):
        """Share of members whose core (mass) region covers each cell."""
        return self.core_masks(mass).mean(axis=0)

    def _band(self, lat, lon):
        """Cells 0.5-2x the source's distance from the receptor: their bearing offsets (deg) from the
        source bearing, and the source distance (m)."""
        coslat = math.cos(math.radians(self.lat0))
        xs = (lon - self.lon0) * DEG_M * coslat
        ys = (lat - self.lat0) * DEG_M
        d_s = math.hypot(xs, ys)
        b_s = math.degrees(math.atan2(xs, ys))
        n = 2 * self.half_n + 1
        jj, ii = np.mgrid[0:n, 0:n]
        x = (ii - self.half_n) * self.cell_m
        y = (jj - self.half_n) * self.cell_m
        d = np.hypot(x, y)
        band = (d >= 0.5 * d_s) & (d <= 2.0 * d_s)
        off = (np.degrees(np.arctan2(x, y)) - b_s + 180.0) % 360.0 - 180.0
        return band, off, d_s

    def source_probability(self, lat, lon, central=0.8):
        """Share of members for which the source's bearing lies inside the central `central` share
        (by footprint weight) of the directions the air came from, at the source's distance
        (cells 0.5-2x as far from the receptor as the source)."""
        band, off, _ = self._band(lat, lon)
        lo_q, hi_q = (1 - central) / 2, (1 + central) / 2
        hits = []
        for m in self.members:
            w = m.grid[band]
            if w.sum() <= 0:
                hits.append(False)
                continue
            o = off[band]
            # weighted circular centre, then offsets relative to it (avoids the +-180 wrap)
            c = math.degrees(math.atan2(np.sum(w * np.sin(np.radians(o))), np.sum(w * np.cos(np.radians(o)))))
            rel = (o - c + 180.0) % 360.0 - 180.0
            order = np.argsort(rel)
            cw = np.cumsum(w[order]) / w.sum()
            q_lo = rel[order][np.searchsorted(cw, lo_q)]
            q_hi = rel[order][min(np.searchsorted(cw, hi_q), len(cw) - 1)]
            src_rel = (0.0 - c + 180.0) % 360.0 - 180.0          # the source sits at offset 0
            hits.append(q_lo <= src_rel <= q_hi)
        return float(np.mean(hits))

    def sector_share(self, lat, lon, half_deg=22.5):
        """Mean over members of the footprint weight within +-half_deg of the source bearing, as a
        share of all weight 0.5-2x the source's distance away."""
        band, off, _ = self._band(lat, lon)
        sec = band & (np.abs(off) <= half_deg)
        vals = [m.grid[sec].sum() / m.grid[band].sum() for m in self.members if m.grid[band].sum() > 0]
        return float(np.mean(vals)) if vals else 0.0

    def to_geojson(self, min_agreement=0.1, mass=0.5):
        """Cells with agreement >= min_agreement as square polygons (properties: agreement, mean weight)."""
        agr, mean = self.agreement(mass), self.mean_footprint()
        coslat = math.cos(math.radians(self.lat0))
        dlat, dlon = self.cell_m / DEG_M, self.cell_m / (DEG_M * coslat)
        feats = []
        for j, i in zip(*np.nonzero(agr >= min_agreement)):
            la = self.lat0 + (j - self.half_n) * dlat
            lo = self.lon0 + (i - self.half_n) * dlon
            ring = [[lo - dlon / 2, la - dlat / 2], [lo + dlon / 2, la - dlat / 2], [lo + dlon / 2, la + dlat / 2],
                    [lo - dlon / 2, la + dlat / 2], [lo - dlon / 2, la - dlat / 2]]
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                          "properties": {"agreement": round(float(agr[j, i]), 3), "weight": float(mean[j, i])}})
        return {"type": "FeatureCollection", "features": feats,
                "properties": {"members": len(self.members), "sigma_dir_deg": self.sigma_dir_deg,
                               "sigma_speed": self.sigma_speed, "core_mass": mass, "cell_m": self.cell_m}}


_ENS = {}


def _member(i):
    a = _ENS
    dth, fsp = a["perturb"][i]
    cfg = ls.LSConfig(**{**a["cfg"].__dict__, "seed": a["cfg"].seed + 104729 * (i + 1)})
    res = ls.run_ls(PerturbedMet(a["met"], dth, fsp), a["lat"], a["lon"], a["t"], hours=a["hours"],
                    n_particles=a["n"], start_heights_m=a["heights"], direction="backward", cfg=cfg)
    n = 2 * a["half_n"] + 1
    grid = np.zeros((n, n))
    coslat = math.cos(math.radians(a["lat"]))
    for (_t, la, lo, w0, _z) in res.touchdowns:
        j = int(round((la - a["lat"]) * DEG_M / a["cell_m"])) + a["half_n"]
        k = int(round((lo - a["lon"]) * DEG_M * coslat / a["cell_m"])) + a["half_n"]
        if 0 <= j < n and 0 <= k < n:
            grid[j, k] += 2.0 / max(abs(w0), 1e-3)          # WindTrax weight
    s = grid.sum()
    return Member(dth, fsp, grid / s if s > 0 else grid, len(res.touchdowns))


def run_ensemble(met, lat, lon, start_time_utc, hours=6.0, members=12, n_particles=1000,
                 start_heights_m=(10.0,), sigma_dir_deg=20.0, sigma_speed=0.3,
                 cell_m=200.0, half_width_km=15.0, cfg: Optional[ls.LSConfig] = None,
                 workers: Optional[int] = None, seed=2026) -> EnsembleResult:
    """Backward LS ensemble around a receptor. Member 0 is the unperturbed wind."""
    cfg = cfg or ls.LSConfig(dt_adv=60.0, record_every=10 ** 9)
    rng = np.random.default_rng(seed)
    perturb = [(0.0, 1.0)] + [(float(rng.normal(0.0, sigma_dir_deg)), float(math.exp(rng.normal(0.0, sigma_speed))))
                              for _ in range(members - 1)]
    half_n = int(round(half_width_km * 1000.0 / cell_m))
    _ENS.clear()
    _ENS.update(met=met, lat=lat, lon=lon, t=start_time_utc, hours=hours, n=n_particles, heights=start_heights_m,
                cfg=cfg, perturb=perturb, half_n=half_n, cell_m=cell_m)
    workers = workers or max(1, min(4, (os.cpu_count() or 2) - 1))
    if workers > 1:
        with mp.get_context("fork").Pool(min(workers, members)) as pool:
            mem = pool.map(_member, range(members))
    else:
        mem = [_member(i) for i in range(members)]
    _ENS.clear()
    return EnsembleResult(mem, lat, lon, cell_m, half_n, sigma_dir_deg, sigma_speed)
