"""
Terrain helpers for the LS model (2026-10-22, batch-2 items 9-10).

DemGrid            - vectorized ground elevation + gradient from a lat/lon GeoTIFF (default: the 0.01 deg
                     Alberta DEM already used by backtraj_core, AB_winds/data/Alberta_dem_1km.tif).
GriddedWindField   - a site's own hourly 10 m wind field (e.g. the Howe Sound 250 m LiDAR terrain-balanced
                     field from drafts/squamish_winds/dsmodel.py), bilinear in space, linear in time.
TerrainFieldCorrection
                   - plugs a GriddedWindField into HRDPSMet(correction=...): inside the field's coverage the
                     10 m wind is replaced by the terrain field, and the same change is applied with height,
                     fading linearly to zero at 120 m (terrain steering is a near-surface effect). Outside
                     coverage HRDPS is untouched. Stability (u*, L) is then computed from the corrected
                     wind by HRDPSMet as usual. Not chained with StationCorrection: that correction's
                     residuals are relative to raw HRDPS.

Measured value so far (our own tests): the LiDAR terrain balance improved Howe Sound direction a lot
(Squamish Airport 78 -> 91% within 45 deg) but not Hinton (84/80% vs 81/80%), so use it where it was
shown to help.
"""
import math
from typing import Optional

import numpy as np

DEFAULT_DEM = "/opt/airquality/github/AB_winds/data/Alberta_dem_1km.tif"


class DemGrid:
    """Ground elevation (m ASL) on a regular lat/lon grid; NaN outside it."""

    def __init__(self, path: str = DEFAULT_DEM):
        import rasterio
        with rasterio.open(path) as r:
            self.z = r.read(1).astype(np.float32)
            nod = r.nodata
            tr = r.transform
        if nod is not None:
            self.z[self.z == nod] = np.nan
        self.lon0, self.dlon = tr.c, tr.a          # top-left corner, + east
        self.lat0, self.dlat = tr.f, tr.e          # top-left corner, dlat < 0 (north-up)
        self.ny, self.nx = self.z.shape

    def elevation(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        fx = (lon - self.lon0) / self.dlon - 0.5
        fy = (lat - self.lat0) / self.dlat - 0.5
        ok = (fx >= 0) & (fy >= 0) & (fx < self.nx - 1) & (fy < self.ny - 1)
        i = np.where(ok, fy, 0).astype(int)
        j = np.where(ok, fx, 0).astype(int)
        a = np.where(ok, fx - j, 0)
        b = np.where(ok, fy - i, 0)
        z = (self.z[i, j] * (1 - a) * (1 - b) + self.z[i, j + 1] * a * (1 - b)
             + self.z[i + 1, j] * (1 - a) * b + self.z[i + 1, j + 1] * a * b)
        return np.where(ok, z, np.nan)


class GriddedWindField:
    """Hourly 10 m wind on a regular lat/lon grid. U, V: arrays [time, lat, lon] (m/s, toward east/north);
    lats ascending, lons ascending; times: sequence of timezone-aware datetimes."""

    def __init__(self, lats, lons, times, U, V):
        self.lats = np.asarray(lats, dtype=float)
        self.lons = np.asarray(lons, dtype=float)
        self.t = np.array([t.timestamp() for t in times])
        self.U = np.asarray(U, dtype=np.float32)
        self.V = np.asarray(V, dtype=np.float32)

    def _space(self, A, lat, lon):
        fy = np.interp(lat, self.lats, np.arange(self.lats.size), left=np.nan, right=np.nan)
        fx = np.interp(lon, self.lons, np.arange(self.lons.size), left=np.nan, right=np.nan)
        ok = np.isfinite(fx) & np.isfinite(fy)
        i = np.clip(np.where(ok, fy, 0).astype(int), 0, self.lats.size - 2)
        j = np.clip(np.where(ok, fx, 0).astype(int), 0, self.lons.size - 2)
        a = np.where(ok, fx - j, 0)
        b = np.where(ok, fy - i, 0)
        v = (A[i, j] * (1 - a) * (1 - b) + A[i, j + 1] * a * (1 - b)
             + A[i + 1, j] * (1 - a) * b + A[i + 1, j + 1] * a * b)
        return np.where(ok, v, np.nan)

    def __call__(self, t, lat, lon):
        ts = t.timestamp()
        if ts < self.t[0] or ts > self.t[-1]:
            n = np.full(np.shape(lat), np.nan)
            return n, n
        k = int(np.clip(np.searchsorted(self.t, ts) - 1, 0, self.t.size - 2))
        w = (ts - self.t[k]) / max(self.t[k + 1] - self.t[k], 1.0)
        u = self._space(self.U[k], lat, lon) * (1 - w) + self._space(self.U[k + 1], lat, lon) * w
        v = self._space(self.V[k], lat, lon) * (1 - w) + self._space(self.V[k + 1], lat, lon) * w
        return u, v


class TerrainFieldCorrection:
    """HRDPSMet(correction=TerrainFieldCorrection(field)): replace the 10 m wind by the site's terrain field
    where it has data; apply the same vector change above 10 m fading linearly to none at top_m (120 m)."""
    n_obs = 1      # run_dispersion reports "station-corrected" when n_obs > 0; harmless for a terrain field

    def __init__(self, field: GriddedWindField, top_m: float = 120.0):
        self.field = field
        self.top = top_m

    def __call__(self, t, lat, lon, levels):
        tu, tv = self.field(t, lat, lon)
        u10, v10 = levels[10.0]
        du = np.where(np.isfinite(tu), tu - u10, 0.0)
        dv = np.where(np.isfinite(tv), tv - v10, 0.0)
        out = {}
        for h, (u, v) in levels.items():
            f = max(0.0, 1.0 - (h - 10.0) / (self.top - 10.0)) if h >= 10.0 else 1.0
            out[h] = (u + f * du, v + f * dv)
        return out


def brunt_vaisala_stable(ustar, L, z, phi_h):
    """N (1/s) in the stable surface layer from MOST: N^2 = u*^2 phi_h(z/L) / (kappa^2 z L), L > 0."""
    kappa = 0.4
    z = np.maximum(z, 2.0)
    n2 = ustar ** 2 * phi_h / (kappa ** 2 * z * np.maximum(L, 1e-3))
    return np.sqrt(np.maximum(n2, 1e-8))
