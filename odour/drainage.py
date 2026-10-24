"""
Cold-air drainage term for HRDPS winds in a valley (2026-10-24, Hinton first).

HRDPS (2.5 km) can't make night-time drainage: cold air sliding down the side
slopes and along the valley floor. Where a precomputed direction field exists
(data/drainage_<area>.npz, built by drafts/hinton_terrain/make_drainage_field.py
from 1 m LiDAR), blend it in when HRDPS itself shows a ground inversion and
light wind:

    S = T40 - T2 (K)     V = |HRDPS 10 m wind| (km/h)
    w = clip(S / S0, 0, 1) * exp(-(V / VC)^2)
    wind = (1 - w) * HRDPS + w * VD * d      d = unit drainage direction

Applied in full at 10 m, 60% at 40 m, 20% at 80 m, none at 120 m (a shallow
layer). Hinton check (held-out Sep 16 - Oct 23, direction within 45 deg at
Drinnan + Hillcrest): all hours 67 -> 70%, night 67 -> 73%, inversion hours
65 -> 71%. It refines calm-night HRDPS; it can't overrule a confidently wrong
forecast wind (w is small then).
"""

import glob
import os

import numpy as np
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
FIELD_GLOB = os.path.join(os.path.dirname(HERE), "data", "drainage_*.npz")
LEVEL_SHARE = {10.0: 1.0, 40.0: 0.6, 80.0: 0.2, 120.0: 0.0}


class Drainage:
    def __init__(self, path):
        z = np.load(path)
        self.name = str(z["name"])
        self.de, self.dn = z["de"], z["dn"]
        self.x0, self.y1, self.res = float(z["x0"]), float(z["y1"]), float(z["res"])
        self.S0, self.VC, self.VD = float(z["S0"]), float(z["VC"]) / 3.6, float(z["VD"]) / 3.6   # m/s
        self.to_utm = Transformer.from_crs(4326, int(z["epsg"]), always_xy=True)
        self.ny, self.nx = self.de.shape

    def _cells(self, lat, lon):
        x, y = self.to_utm.transform(np.asarray(lon, float), np.asarray(lat, float))
        j = np.floor((x - self.x0) / self.res).astype(int)
        i = np.floor((self.y1 - y) / self.res).astype(int)
        inside = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx) & np.isfinite(x) & np.isfinite(y)
        return np.where(inside, i, 0), np.where(inside, j, 0), inside

    def covers(self, lat, lon):
        return bool(self._cells(np.array([lat]), np.array([lon]))[2][0])

    def __call__(self, lat, lon, lev, T2, T40):
        i, j, inside = self._cells(lat, lon)
        if not inside.any() or T2 is None or T40 is None:
            return lev
        u10, v10 = lev[10.0]
        S = np.nan_to_num(np.asarray(T40) - np.asarray(T2), nan=0.0)
        w = np.clip(S / self.S0, 0.0, 1.0) * np.exp(-(np.hypot(u10, v10) / self.VC) ** 2)
        w = np.where(inside, np.nan_to_num(w), 0.0)
        du, dv = self.VD * self.de[i, j], self.VD * self.dn[i, j]
        out = {}
        for h, (u, v) in lev.items():
            a = w * LEVEL_SHARE.get(h, 0.0)
            out[h] = ((1 - a) * u + a * du, (1 - a) * v + a * dv)
        return out


def for_point(lat, lon):
    """The drainage field covering this point, or None."""
    for p in sorted(glob.glob(FIELD_GLOB)):
        d = Drainage(p)
        if d.covers(lat, lon):
            return d
    return None
