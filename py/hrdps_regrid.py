# hrdps_regrid.py
#
# HRDPS continental GRIB2 files are on a ROTATED lat/lon grid (RLatLon0.0225):
# rows/columns do not follow parallels/meridians, rows are stored south->north,
# and UGRD/VGRD are relative to the rotated grid axes (uvRelativeToGrid=1).
#
# Every consumer of the ab_met JSON files (index.html wind animation,
# odour/backtraj_core.py, odour/ls_model.py, the sit-rep wind_shear modules)
# indexes the grid as a plain north-up regular lat/lon grid:
#     i = (lon - lo1) / dx ,  j = (la1 - lat) / dy
# This module resamples the rotated grid onto exactly that kind of grid and
# rotates the wind components to earth-relative (u = eastward, v = northward),
# so the stored files finally mean what every reader assumes they mean.
#
# Before 2026-10-13 the raw rotated arrays were stored with a bounding-box
# header, which put every wind value 100-900 km from where it belonged.

import numpy as np
import eccodes

# Target grid: regular lat/lon, row 0 = north edge. ~2.5 km spacing at 54N.
LAT_N, LAT_S = 62.0, 48.0
LON_W, LON_E = -125.0, -88.0
DLAT = 0.0225
DLON = 0.0375

GRID_VERSION = 2  # stored in meta; files without it are the old, misplaced layout


def _xyz(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _latlon(xyz):
    x, y, z = xyz
    return np.degrees(np.arcsin(np.clip(z, -1, 1))), np.degrees(np.arctan2(y, x))


class RotatedGrid:
    """Geometry of one HRDPS rotated lat/lon GRIB2 grid, read from the file itself."""

    def __init__(self, grib_path):
        with open(grib_path, "rb") as f:
            g = eccodes.codes_grib_new_from_file(f)
            try:
                get = lambda k: eccodes.codes_get(g, k)
                if get("gridType") != "rotated_ll":
                    raise ValueError(f"expected rotated_ll grid, got {get('gridType')}")
                if get("iScansNegatively") != 0 or get("jScansPositively") != 1:
                    raise ValueError("unexpected HRDPS scanning mode")
                if float(get("angleOfRotationInDegrees")) != 0.0:
                    raise ValueError("non-zero angle of rotation not supported")
                self.uv_grid_relative = bool(get("uvRelativeToGrid"))
                self.rlat0 = float(get("latitudeOfFirstGridPointInDegrees"))
                self.rlon0 = float(get("longitudeOfFirstGridPointInDegrees"))
                self.di = float(get("iDirectionIncrementInDegrees"))
                self.dj = float(get("jDirectionIncrementInDegrees"))
                self.ni = int(get("Ni"))
                self.nj = int(get("Nj"))
                sp_lat = float(get("latitudeOfSouthernPoleInDegrees"))
                sp_lon = float(get("longitudeOfSouthernPoleInDegrees"))
            finally:
                eccodes.codes_release(g)

        # Rotated north pole, and the matrix taking geographic xyz -> rotated xyz.
        np_lat, np_lon = -sp_lat, sp_lon - 180.0
        a, b = np.radians(np_lon), np.radians(np_lat - 90.0)
        rz = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])
        ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        self.R = ry @ rz

    def to_rotated(self, lat, lon):
        shp = np.shape(lat)
        v = self.R @ _xyz(np.ravel(lat), np.ravel(lon))
        rlat, rlon = _latlon(v)
        # GRIB's rotated longitude origin is 180 deg from this rotation's.
        return rlat.reshape(shp), (rlon + 180.0).reshape(shp)

    def to_geographic(self, rlat, rlon):
        shp = np.shape(rlat)
        v = self.R.T @ _xyz(np.ravel(rlat), np.ravel(rlon) - 180.0)
        lat, lon = _latlon(v)
        return lat.reshape(shp), lon.reshape(shp)

    def fractional_index(self, lat, lon):
        """(fj, fi) into the GRIB array (row 0 = first/southern row)."""
        rlat, rlon = self.to_rotated(lat, lon)
        fi = ((rlon - self.rlon0) % 360.0) / self.di
        # a hair west of the first column wraps to ~360 deg; bring it back to ~0
        full_turn = 360.0 / self.di
        fi = np.where(fi > full_turn - 0.5, fi - full_turn, fi)
        fj = (rlat - self.rlat0) / self.dj
        return fj, fi


class Regridder:
    """Bilinear resampling of a rotated HRDPS grid onto the regular target grid."""

    def __init__(self, grib_path):
        self.src = RotatedGrid(grib_path)
        self.lats = LAT_N - DLAT * np.arange(int(round((LAT_N - LAT_S) / DLAT)) + 1)
        self.lons = LON_W + DLON * np.arange(int(round((LON_E - LON_W) / DLON)) + 1)
        lon2, lat2 = np.meshgrid(self.lons, self.lats)

        fj, fi = self.src.fractional_index(lat2, lon2)
        self.inside = (fi >= 0) & (fi <= self.src.ni - 1) & (fj >= 0) & (fj <= self.src.nj - 1)
        fi = np.clip(fi, 0, self.src.ni - 1.000001)
        fj = np.clip(fj, 0, self.src.nj - 1.000001)
        self.i0, self.j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
        self.wi, self.wj = fi - self.i0, fj - self.j0

        # Angle of the rotated grid's x-axis, counter-clockwise from true east,
        # at each target point: step +di along rotated longitude and measure it.
        rlat, rlon = self.src.to_rotated(lat2, lon2)
        lat_b, lon_b = self.src.to_geographic(rlat, rlon + self.src.di)
        dlon = ((lon_b - lon2 + 180.0) % 360.0) - 180.0
        self.alpha = np.arctan2(lat_b - lat2, dlon * np.cos(np.radians(lat2)))

    @property
    def grid(self):
        return {
            "lo1": float(self.lons[0]),
            "la1": float(self.lats[0]),
            "dx": DLON,
            "dy": DLAT,
            "nx": int(len(self.lons)),
            "ny": int(len(self.lats)),
        }

    def resample(self, arr):
        """arr: 2-D (nj, ni) GRIB-order array -> 2-D (ny, nx) north-up float32, NaN outside."""
        a = np.asarray(arr, dtype=np.float64)
        if a.shape != (self.src.nj, self.src.ni):
            raise ValueError(f"array shape {a.shape} != grid {(self.src.nj, self.src.ni)}")
        j0, i0, wj, wi = self.j0, self.i0, self.wj, self.wi
        out = (a[j0, i0] * (1 - wi) * (1 - wj) + a[j0, i0 + 1] * wi * (1 - wj)
               + a[j0 + 1, i0] * (1 - wi) * wj + a[j0 + 1, i0 + 1] * wi * wj)
        out[~self.inside] = np.nan
        return out.astype(np.float32)

    def earth_relative(self, u_grid, v_grid):
        """Already-resampled grid-relative u/v -> eastward/northward components."""
        if not self.src.uv_grid_relative:
            return u_grid, v_grid
        c, s = np.cos(self.alpha), np.sin(self.alpha)
        u = u_grid * c - v_grid * s
        v = u_grid * s + v_grid * c
        return u.astype(np.float32), v.astype(np.float32)


_CACHE = {}


def regridder_for(grib_path):
    """One Regridder per distinct grid definition (all HRDPS continental files share one)."""
    src = RotatedGrid(grib_path)
    key = (src.rlat0, src.rlon0, src.di, src.dj, src.ni, src.nj, tuple(np.round(src.R.ravel(), 12)))
    if key not in _CACHE:
        _CACHE[key] = Regridder(grib_path)
    return _CACHE[key]
