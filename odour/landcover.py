"""
Surface roughness (z0) from land cover, for the LS model's u* and
near-surface wind profile (micromet.surface_layer_from_profile).

Data: ESA WorldCover 10 m (2021, v200), read straight from its public
cloud-optimised GeoTIFFs on AWS - only the window around the run is
fetched, from a reduced-resolution overview (~0.8 s for a 50 km window
around Edmonton). No key, no bulk download.

z0 per class (m), typical literature values (Wieringa 1992; Stull 1988;
AERMET land-use tables), with seasonal crop/grass values for Alberta:
snow-covered or harvested fields are far smoother than a July canola crop.
Pixels are blended by averaging ln(z0) over ~1 km, the standard way to
aggregate roughness (an arithmetic mean lets a few trees dominate).

Not modelled: displacement height over forest and downtown (wind starts
at roughly 2/3 of canopy/building height). That matters only very close
to the surface and is left for later.
"""

import math
import os
from typing import Optional

import numpy as np

CACHE_DIR = "/opt/airquality/dsai_data/landcover_cache"
WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")

# WorldCover class -> z0 (m) by season: (summer May-Sep, shoulder Apr/Oct, winter Nov-Mar)
Z0_TABLE = {
    10: (0.8, 0.8, 0.7),        # tree cover (mostly conifer/aspen mix)
    20: (0.2, 0.15, 0.1),       # shrubland
    30: (0.05, 0.03, 0.01),     # grassland
    40: (0.1, 0.03, 0.01),      # cropland (standing crop -> stubble -> snow)
    50: (1.0, 1.0, 1.0),        # built-up
    60: (0.01, 0.01, 0.005),    # bare / sparse vegetation
    70: (0.001, 0.001, 0.001),  # snow and ice
    80: (0.0002, 0.0002, 0.001),  # open water (frozen in winter)
    90: (0.05, 0.03, 0.01),     # herbaceous wetland
    95: (0.5, 0.5, 0.5),        # mangroves (not in Alberta)
    100: (0.01, 0.01, 0.005),   # moss and lichen
}
DEFAULT_Z0 = 0.1


def season_index(month: int) -> int:
    if 5 <= month <= 9:
        return 0
    if month in (4, 10):
        return 1
    return 2


def _tile_name(lat_floor3: int, lon_floor3: int) -> str:
    ns = "N" if lat_floor3 >= 0 else "S"
    ew = "E" if lon_floor3 >= 0 else "W"
    return f"{ns}{abs(lat_floor3):02d}{ew}{abs(lon_floor3):03d}"


def _read_worldcover(lat0, lat1, lon0, lon1, res_deg, res_lon=None):
    """Class array covering the bbox at ~res_deg, stitched across 3x3-degree
    tiles. Returns (classes uint8 array [rows north->south, cols west->east])."""
    import rasterio
    from rasterio.windows import from_bounds
    res_lon = res_lon or res_deg
    ny = max(int(round((lat1 - lat0) / res_deg)), 1)
    nx = max(int(round((lon1 - lon0) / res_lon)), 1)
    out = np.zeros((ny, nx), dtype=np.uint8)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_TIMEOUT="30"):
        for tla in range(int(math.floor(lat0 / 3) * 3), int(math.floor(lat1 / 3) * 3) + 1, 3):
            for tlo in range(int(math.floor(lon0 / 3) * 3), int(math.floor(lon1 / 3) * 3) + 1, 3):
                a0, a1 = max(lat0, tla), min(lat1, tla + 3)
                o0, o1 = max(lon0, tlo), min(lon1, tlo + 3)
                if a1 <= a0 or o1 <= o0:
                    continue
                r0 = int(round((lat1 - a1) / res_deg))
                r1 = int(round((lat1 - a0) / res_deg))
                c0 = int(round((o0 - lon0) / res_lon))
                c1 = int(round((o1 - lon0) / res_lon))
                if r1 <= r0 or c1 <= c0:
                    continue
                url = "/vsicurl/" + WC_URL.format(tile=_tile_name(tla, tlo))
                with rasterio.open(url) as src:
                    win = from_bounds(o0, a0, o1, a1, src.transform)
                    out[r0:r1, c0:c1] = src.read(1, window=win, out_shape=(r1 - r0, c1 - c0))
    return out


class RoughnessMap:
    """z0 grid around a site. Call as z0_fn(lat, lon) -> z0 array."""

    def __init__(self, lat: float, lon: float, radius_km: float = 30.0, month: int = 7,
                 cell_m: float = 250.0, smooth_m: float = 1000.0, use_cache: bool = True):
        self.month = month
        dlat = radius_km * 1000.0 / 111320.0
        dlon = dlat / math.cos(math.radians(lat))
        self.lat0, self.lat1 = lat - dlat, lat + dlat
        self.lon0, self.lon1 = lon - dlon, lon + dlon
        self.res = cell_m / 111320.0                 # cell size in latitude degrees
        self.res_lon = self.res / math.cos(math.radians(lat))
        key = f"{lat:.3f}_{lon:.3f}_{radius_km:.0f}_{cell_m:.0f}_v2"
        path = os.path.join(CACHE_DIR, f"wc_{key}.npz")
        classes_fine = None
        if use_cache and os.path.exists(path):
            classes_fine = np.load(path)["classes"]
        if classes_fine is None:
            # read at ~1/4 of the cell size so each cell blends ~16 pixels
            classes_fine = _read_worldcover(self.lat0, self.lat1, self.lon0, self.lon1,
                                            self.res / 4, self.res_lon / 4)
            if use_cache:
                os.makedirs(CACHE_DIR, exist_ok=True)
                np.savez_compressed(path, classes=classes_fine)
        self.classes_fine = classes_fine
        self.z0 = self._build_grid(classes_fine, smooth_m, cell_m)

    def _build_grid(self, cls, smooth_m, cell_m):
        si = season_index(self.month)
        lut = np.full(256, DEFAULT_Z0)
        for c, vals in Z0_TABLE.items():
            lut[c] = vals[si]
        lnz = np.log(lut[cls])
        # block-average ln z0 (4x4 fine pixels -> one cell), then a box
        # filter of ~smooth_m to represent the upwind fetch
        ny, nx = lnz.shape[0] // 4, lnz.shape[1] // 4
        lnz = lnz[:ny * 4, :nx * 4].reshape(ny, 4, nx, 4).mean(axis=(1, 3))
        k = max(int(round(smooth_m / cell_m)), 1)
        if k > 1:
            pad = np.pad(lnz, k // 2, mode="edge")
            cs = pad.cumsum(0).cumsum(1)
            cs = np.pad(cs, ((1, 0), (1, 0)))
            lnz = (cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k])[:ny, :nx] / (k * k)
        return np.exp(lnz)

    def class_fractions(self):
        u, c = np.unique(self.classes_fine, return_counts=True)
        return {int(a): float(b) / float(c.sum()) for a, b in zip(u, c)}

    def __call__(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        ny, nx = self.z0.shape
        j = np.floor((self.lat1 - lat) / self.res).astype(int)
        i = np.floor((lon - self.lon0) / self.res_lon).astype(int)
        inside = (j >= 0) & (j < ny) & (i >= 0) & (i < nx)
        out = np.full(lat.shape, DEFAULT_Z0)
        out[inside] = self.z0[j[inside], i[inside]]
        return out
