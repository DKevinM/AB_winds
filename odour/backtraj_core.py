import os, re, json, gzip
import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

EARTH_R = 6371000.0  # meters


# ---------------------------
# Helpers
# ---------------------------

def parse_iso_z(s: str) -> dt.datetime:
    # expects "....Z"
    s = s.replace("Z", "")
    # Python can parse ISO without Z
    return dt.datetime.fromisoformat(s)

def clamp(x, a, b):
    return a if x < a else b if x > b else x

def lon_wrap(lon):
    # keep [-180,180)
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon

def datetime_floor_to_hour(t: dt.datetime) -> dt.datetime:
    return t.replace(minute=0, second=0, microsecond=0)


# ---------------------------
# Data model
# ---------------------------

@dataclass(frozen=True)
class Grid:
    lo1: float   # west lon (deg, -180..180)
    la1: float   # north lat (deg)
    dx: float    # lon spacing (deg, positive)
    dy: float    # lat spacing (deg, positive)
    nx: int
    ny: int

    def ij_from_latlon(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Earth-style grid:
          i increases east from lo1
          j increases south from la1
        """
        i = (lon - self.lo1) / self.dx
        j = (self.la1 - lat) / self.dy
        return i, j

    def bounds(self):
        lon_min = self.lo1
        lon_max = self.lo1 + self.dx * (self.nx - 1)
        lat_max = self.la1
        lat_min = self.la1 - self.dy * (self.ny - 1)
        return lat_min, lat_max, lon_min, lon_max


@dataclass
class MetSnapshot:
    valid: dt.datetime
    run: dt.datetime
    lead: int
    grid: Grid
    fields: Dict[str, np.ndarray]  # key -> (ny,nx) float32


# ---------------------------
# Met store (loads your *.json.gz)
# ---------------------------

class MetStore:
    """
    Expects each gz json to look like:
    {
      "meta": {"run": "...Z", "valid": "...Z", "lead": 6},
      "grid": {"lo1":..., "la1":..., "dx":..., "dy":..., "nx":..., "ny":...},
      "fields": {
         "u10": [... ny*nx ...],
         "v10": [...],
         "u40": [...],
         "v40": [...],
         "u80": [...],
         "v80": [...],
         "u120":[...],
         "v120":[...],
         "hpbl":[...],   # meters
         "rh2m":[...],   # %
         "dpt2m":[...]   # K or C (doesn't matter for trajectories)
      }
    }
    Field names are flexible; we normalize a few common patterns.
    """

    def __init__(self, folder: str):
        self.folder = folder
        self.snaps: List[MetSnapshot] = []
        self._load_all()

        if not self.snaps:
            raise RuntimeError(f"No met files found in {folder}")

        # ensure sorted by valid time
        self.snaps.sort(key=lambda s: s.valid)
        self.grid = self.snaps[0].grid

    def _load_all(self):
        for fn in os.listdir(self.folder):
            if not (fn.endswith(".json.gz") or fn.endswith(".json.gzip")):
                continue
            path = os.path.join(self.folder, fn)
            snap = self._load_one(path)
            self.snaps.append(snap)

    def _load_one(self, path: str) -> MetSnapshot:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            obj = json.load(f)

        meta = obj["meta"]
        gridj = obj["grid"]
        fieldsj = obj["fields"]

        grid = Grid(
            lo1=float(gridj["lo1"]),
            la1=float(gridj["la1"]),
            dx=float(gridj["dx"]),
            dy=float(gridj["dy"]),
            nx=int(gridj["nx"]),
            ny=int(gridj["ny"])
        )

        def to_grid(arr_1d):
            a = np.array(arr_1d, dtype=np.float32)
            if a.size != grid.nx * grid.ny:
                raise ValueError(f"Grid size mismatch: got {a.size}, expected {grid.nx*grid.ny}")
            return a.reshape((grid.ny, grid.nx))

        # normalize keys (support a few variants)
        norm = {}
        for k, v in fieldsj.items():
            kk = k.lower().strip()

            # common normalizations
            kk = kk.replace("ugrd", "u").replace("vgrd", "v")
            kk = kk.replace("agl-", "")
            kk = kk.replace("m", "m")  # no-op but keeps intent

            # make canonical keys for winds
            # accept u10, u10m, u_10, etc.
            m = re.match(r"^(u|v)\D*(10|40|80|120)\D*$", kk)
            if m:
                kk = f"{m.group(1)}{m.group(2)}"
            if kk in ("hpbl", "hpblsfc", "pblh", "hpbl_sfc"):
                kk = "hpbl"
            if kk in ("rh2m", "rh2", "rh_2m", "rhagl2m"):
                kk = "rh2m"
            if kk in ("dpt2m", "dpt2", "td2m", "dpt_2m", "dptagl2m"):
                kk = "dpt2m"

            norm[kk] = to_grid(v)

        return MetSnapshot(
            valid=parse_iso_z(meta["valid"]),
            run=parse_iso_z(meta["run"]),
            lead=int(meta.get("lead", meta.get("forecastTime", 0))),
            grid=grid,
            fields=norm
        )

    def _bracket(self, t: dt.datetime) -> Tuple[MetSnapshot, MetSnapshot, float]:
        """
        Find snapshots s0,s1 such that s0.valid <= t <= s1.valid.
        Returns (s0,s1,alpha) where alpha in [0,1] for time interpolation.
        If outside range, clamps to ends (alpha=0).
        """
        snaps = self.snaps
        if t <= snaps[0].valid:
            return snaps[0], snaps[0], 0.0
        if t >= snaps[-1].valid:
            return snaps[-1], snaps[-1], 0.0

        # binary search
        lo, hi = 0, len(snaps) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if snaps[mid].valid <= t:
                lo = mid
            else:
                hi = mid

        s0, s1 = snaps[lo], snaps[hi]
        dt_total = (s1.valid - s0.valid).total_seconds()
        alpha = 0.0 if dt_total <= 0 else (t - s0.valid).total_seconds() / dt_total
        alpha = clamp(alpha, 0.0, 1.0)
        return s0, s1, alpha

    def _bilinear(self, grid: Grid, field: np.ndarray, lat: float, lon: float) -> float:
        i, j = grid.ij_from_latlon(lat, lon)

        # clamp inside grid cell range
        i = clamp(i, 0.0, grid.nx - 1.000001)
        j = clamp(j, 0.0, grid.ny - 1.000001)

        i0 = int(math.floor(i))
        j0 = int(math.floor(j))
        i1 = min(i0 + 1, grid.nx - 1)
        j1 = min(j0 + 1, grid.ny - 1)

        fi = i - i0
        fj = j - j0

        v00 = float(field[j0, i0])
        v10 = float(field[j0, i1])
        v01 = float(field[j1, i0])
        v11 = float(field[j1, i1])

        v0 = v00 * (1 - fi) + v10 * fi
        v1 = v01 * (1 - fi) + v11 * fi
        return v0 * (1 - fj) + v1 * fj

    def sample(self, t: dt.datetime, lat: float, lon: float, z_m: float) -> Dict[str, float]:
        """
        Returns sampled met at (t,lat,lon,z).
        Winds are returned as u,v in m/s for nearest AGL layer among 10/40/80/120.
        Also returns hpbl (m) if available.
        """
        s0, s1, a = self._bracket(t)
        g = s0.grid

        # choose nearest vertical layer
        layers = np.array([10, 40, 80, 120], dtype=float)
        zpick = int(layers[np.argmin(np.abs(layers - z_m))])

        ukey = f"u{zpick}"
        vkey = f"v{zpick}"

        def interp_key(key: str) -> Optional[float]:
            if key not in s0.fields or key not in s1.fields:
                return None
            v0 = self._bilinear(g, s0.fields[key], lat, lon)
            v1 = self._bilinear(g, s1.fields[key], lat, lon)
            return v0 * (1 - a) + v1 * a

        u = interp_key(ukey)
        v = interp_key(vkey)
        hpbl = interp_key("hpbl")

        if u is None or v is None:
            raise KeyError(f"Missing wind fields for layer {zpick}m: need {ukey},{vkey}")

        out = {"u": float(u), "v": float(v), "zlayer": float(zpick)}
        if hpbl is not None:
            out["hpbl"] = float(hpbl)
        return out


# ---------------------------
# Back trajectory model
# ---------------------------

@dataclass
class ParticleState:
    lat: float
    lon: float
    z_m: float

def advect_latlon(lat: float, lon: float, u: float, v: float, dt_s: float) -> Tuple[float, float]:
    """
    Convert (u,v) m/s to (dlat,dlon) degrees over dt.
    """
    lat_rad = math.radians(lat)
    dlat = (v * dt_s) / EARTH_R * (180.0 / math.pi)
    dlon = (u * dt_s) / (EARTH_R * max(1e-8, math.cos(lat_rad))) * (180.0 / math.pi)
    return lat + dlat, lon_wrap(lon + dlon)


def run_back_trajectories(
    met: MetStore,
    start_lat: float,
    start_lon: float,
    start_time_utc: dt.datetime,
    hours: float = 5.0,
    dt_s: int = 60,
    n_particles: int = 200,
    start_heights_m: List[float] = (10.0, 40.0, 80.0),
    horiz_sigma_ms: float = 0.35,   # random wind perturbation (m/s)
    vert_sigma_ms: float = 0.10,    # vertical random walk (m/s equivalent)
    use_pbl_cap: bool = True
):
    """
    Backward Lagrangian ensemble:
      - integrates from start_time_utc backward for 'hours'
      - uses simple stochastic perturbations to represent uncertainty
      - returns centerlines and particle cloud

    Notes:
      - Backward integration is done by flipping wind sign (u,v -> -u,-v).
      - We do RK2 (midpoint) for better stability than Euler.
      - Vertical motion: no resolved w, so we random-walk z with optional PBL cap.
    """
    n_steps = int((hours * 3600) // dt_s)
    t0 = start_time_utc

    rng = np.random.default_rng(12345)

    # outputs
    centerlines = []  # list of dicts: {"z0":..., "track":[(t,lat,lon,z),...]}
    cloud_points = [] # list of (t,lat,lon,z)

    for z0 in start_heights_m:
        # initialize particles
        parts = [ParticleState(start_lat, start_lon, float(z0)) for _ in range(n_particles)]
        track_center = []

        for k in range(n_steps + 1):
            t = t0 - dt.timedelta(seconds=k * dt_s)

            # record center
            lat_c = float(np.mean([p.lat for p in parts]))
            lon_c = float(np.mean([p.lon for p in parts]))
            z_c   = float(np.mean([p.z_m for p in parts]))
            track_center.append((t, lat_c, lon_c, z_c))

            # record cloud (downsample to keep size reasonable)
            if k % 5 == 0:
                for p in parts:
                    cloud_points.append((t, p.lat, p.lon, p.z_m))

            # step particles backward (skip stepping at final record)
            if k == n_steps:
                break

            for p in parts:
                # --- RK2 midpoint step ---
                m1 = met.sample(t, p.lat, p.lon, p.z_m)
                u1 = -m1["u"] + rng.normal(0, horiz_sigma_ms)
                v1 = -m1["v"] + rng.normal(0, horiz_sigma_ms)

                # midpoint prediction
                lat_mid, lon_mid = advect_latlon(p.lat, p.lon, u1, v1, dt_s * 0.5)

                m2 = met.sample(t - dt.timedelta(seconds=dt_s * 0.5), lat_mid, lon_mid, p.z_m)
                u2 = -m2["u"] + rng.normal(0, horiz_sigma_ms)
                v2 = -m2["v"] + rng.normal(0, horiz_sigma_ms)

                lat_new, lon_new = advect_latlon(p.lat, p.lon, u2, v2, dt_s)

                # vertical random walk (simple)
                z_new = p.z_m + rng.normal(0.0, vert_sigma_ms) * dt_s  # meters-ish

                # apply PBL cap if available
                if use_pbl_cap and "hpbl" in m2 and np.isfinite(m2["hpbl"]):
                    hpbl = max(20.0, float(m2["hpbl"]))
                    z_new = clamp(z_new, 2.0, hpbl)
                else:
                    z_new = clamp(z_new, 2.0, 500.0)

                # update
                p.lat, p.lon, p.z_m = lat_new, lon_new, z_new

        centerlines.append({"z0": float(z0), "track": track_center})

    return centerlines, cloud_points


# ---------------------------
# GeoJSON writers (optional)
# ---------------------------

def centerlines_to_geojson(centerlines):
    feats = []
    for c in centerlines:
        coords = [[lon, lat] for (t, lat, lon, z) in c["track"]]
        feats.append({
            "type": "Feature",
            "properties": {"z0_m": c["z0"]},
            "geometry": {"type": "LineString", "coordinates": coords}
        })
    return {"type": "FeatureCollection", "features": feats}

def cloud_to_geojson(cloud_points, every_n: int = 1):
    feats = []
    for idx, (t, lat, lon, z) in enumerate(cloud_points):
        if every_n > 1 and (idx % every_n) != 0:
            continue
        feats.append({
            "type": "Feature",
            "properties": {"t": t.isoformat() + "Z", "z_m": float(z)},
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]}
        })
    return {"type": "FeatureCollection", "features": feats}
