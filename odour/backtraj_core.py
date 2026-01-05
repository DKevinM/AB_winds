# backtraj_core_v2.py
import os, re, json, gzip, math
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
    if s.endswith("Z"):
        s = s[:-1]
    return dt.datetime.fromisoformat(s)

def clamp(x, a, b):
    return a if x < a else b if x > b else x

def lon_wrap(lon):
    # keep [-180,180)
    return (lon + 180.0) % 360.0 - 180.0


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
        Earth/Nullschool-style grid:
          i increases east from lo1
          j increases south from la1
        """
        i = (lon - self.lo1) / self.dx
        j = (self.la1 - lat) / self.dy
        return i, j


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

class MetStoreV2:
    """
    Expects your met files like:
    {
      "meta": {"run": "...Z", "valid": "...Z", "lead": 0},
      "grid": {"lo1":..., "la1":..., "dx":..., "dy":..., "nx":..., "ny":...},
      "fields": {
         "ugrd10": [[[...]]],   # usually (1,ny,nx) or (ny,nx)
         "vgrd10": [[[...]]],
         ...
         "hpbl":  [[[...]]],
         "rh2m":  [[[...]]],
         "dpt2m": [[[...]]]
      }
    }
    """

    def __init__(self, folder: str):
        self.folder = folder
        self.snaps: List[MetSnapshot] = []
        self._load_all()

        if not self.snaps:
            raise RuntimeError(f"No met files found in {folder}")

        self.snaps.sort(key=lambda s: s.valid)
        # assume consistent grid
        self.grid = self.snaps[0].grid

    def _load_all(self):
        for fn in os.listdir(self.folder):
            if not fn.endswith(".json.gz"):
                continue
            path = os.path.join(self.folder, fn)
            self.snaps.append(self._load_one(path))

    def _normalize_key(self, k: str) -> str:
        kk = k.lower().strip()

        # winds: ugrd10, vgrd40, etc -> u10, v40
        m = re.match(r"^(u|v)grd\D*(10|40|80|120)\D*$", kk)
        if m:
            return f"{m.group(1)}{m.group(2)}"

        # sometimes you might have "ugrd10m" etc
        m2 = re.match(r"^(u|v)\D*(10|40|80|120)\D*$", kk)
        if m2:
            return f"{m2.group(1)}{m2.group(2)}"

        # pbl height
        if kk in ("hpbl", "pblh", "hpblsfc", "hpbl_sfc"):
            return "hpbl"

        # RH and dewpoint near-sfc
        if kk in ("rh2m", "rh_2m", "rhagl2m"):
            return "rh2m"
        if kk in ("dpt2m", "dpt_2m", "td2m"):
            return "dpt2m"

        return kk

    def _to_grid(self, arr, grid: Grid) -> np.ndarray:
        """
        Accepts arr shaped:
          - (1,ny,nx)
          - (ny,nx)
          - (ny*nx,)
        Returns (ny,nx) float32.
        """
        a = np.asarray(arr, dtype=np.float32)

        # drop any leading singleton dims: (1,ny,nx) -> (ny,nx)
        a = np.squeeze(a)

        if a.ndim == 2:
            if a.shape != (grid.ny, grid.nx):
                raise ValueError(f"2D field shape mismatch: {a.shape} != {(grid.ny, grid.nx)}")
            return a

        if a.ndim == 1:
            if a.size != grid.ny * grid.nx:
                raise ValueError(f"1D field size mismatch: {a.size} != {grid.ny*grid.nx}")
            return a.reshape((grid.ny, grid.nx))

        raise ValueError(f"Unsupported field dims: ndim={a.ndim}, shape={a.shape}")

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

        norm_fields: Dict[str, np.ndarray] = {}
        for k, v in fieldsj.items():
            kk = self._normalize_key(k)
            norm_fields[kk] = self._to_grid(v, grid)

        return MetSnapshot(
            valid=parse_iso_z(meta["valid"]),
            run=parse_iso_z(meta["run"]),
            lead=int(meta.get("lead", 0)),
            grid=grid,
            fields=norm_fields
        )

    # ----- sampling -----

    def _bracket(self, t: dt.datetime) -> Tuple[MetSnapshot, MetSnapshot, float]:
        snaps = self.snaps
        if t <= snaps[0].valid:
            return snaps[0], snaps[0], 0.0
        if t >= snaps[-1].valid:
            return snaps[-1], snaps[-1], 0.0

        lo, hi = 0, len(snaps) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if snaps[mid].valid <= t:
                lo = mid
            else:
                hi = mid

        s0, s1 = snaps[lo], snaps[hi]
        dt_total = (s1.valid - s0.valid).total_seconds()
        a = 0.0 if dt_total <= 0 else (t - s0.valid).total_seconds() / dt_total
        return s0, s1, float(clamp(a, 0.0, 1.0))

    def _bilinear(self, grid: Grid, field: np.ndarray, lat: float, lon: float) -> float:
        i, j = grid.ij_from_latlon(lat, lon)

        # clamp inside grid
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
        Sample met at (t,lat,lon,z).
        Returns u,v (m/s) at nearest layer among 10/40/80/120 plus hpbl if present.
        """
        s0, s1, a = self._bracket(t)
        g = s0.grid

        layers = np.array([10, 40, 80, 120], dtype=float)
        zpick = int(layers[np.argmin(np.abs(layers - z_m))])

        ukey = f"u{zpick}"
        vkey = f"v{zpick}"

        def interp(key: str) -> Optional[float]:
            if key not in s0.fields or key not in s1.fields:
                return None
            v0 = self._bilinear(g, s0.fields[key], lat, lon)
            v1 = self._bilinear(g, s1.fields[key], lat, lon)
            return v0 * (1 - a) + v1 * a

        u = interp(ukey)
        v = interp(vkey)
        hpbl = interp("hpbl")

        if u is None or v is None:
            raise KeyError(f"Missing winds: need {ukey} and {vkey}")

        out = {"u": float(u), "v": float(v), "zlayer": float(zpick)}
        if hpbl is not None and np.isfinite(hpbl):
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
    met: MetStoreV2,
    start_lat: float,
    start_lon: float,
    start_time_utc: dt.datetime,
    hours: float = 5.0,
    dt_s: int = 60,
    n_particles: int = 200,
    start_heights_m: Tuple[float, ...] = (10.0, 40.0, 80.0),
    horiz_sigma_ms: float = 0.35,
    vert_sigma_ms: float = 0.10,
    use_pbl_cap: bool = True,
    seed: int = 12345
):
    """
    Backward Lagrangian ensemble:
      - integrate from start_time_utc backward for 'hours'
      - RK2 midpoint
      - stochastic u/v perturbations to represent uncertainty
      - vertical random-walk with optional PBL cap
    """
    n_steps = int((hours * 3600) // dt_s)
    rng = np.random.default_rng(seed)

    centerlines = []
    cloud_points = []

    for z0 in start_heights_m:
        parts = [ParticleState(start_lat, start_lon, float(z0)) for _ in range(n_particles)]
        track_center = []

        for k in range(n_steps + 1):
            t = start_time_utc - dt.timedelta(seconds=k * dt_s)

            lat_c = float(np.mean([p.lat for p in parts]))
            lon_c = float(np.mean([p.lon for p in parts]))
            z_c   = float(np.mean([p.z_m for p in parts]))
            track_center.append((t, lat_c, lon_c, z_c))

            # cloud downsample
            if k % 5 == 0:
                for p in parts:
                    cloud_points.append((t, p.lat, p.lon, p.z_m))

            if k == n_steps:
                break

            for p in parts:
                # RK2 midpoint (backward => negate winds)
                m1 = met.sample(t, p.lat, p.lon, p.z_m)
                u1 = -m1["u"] + rng.normal(0, horiz_sigma_ms)
                v1 = -m1["v"] + rng.normal(0, horiz_sigma_ms)

                lat_mid, lon_mid = advect_latlon(p.lat, p.lon, u1, v1, dt_s * 0.5)

                m2 = met.sample(t - dt.timedelta(seconds=dt_s * 0.5), lat_mid, lon_mid, p.z_m)
                u2 = -m2["u"] + rng.normal(0, horiz_sigma_ms)
                v2 = -m2["v"] + rng.normal(0, horiz_sigma_ms)

                lat_new, lon_new = advect_latlon(p.lat, p.lon, u2, v2, dt_s)

                # vertical random walk (meters-ish)
                z_new = p.z_m + rng.normal(0.0, vert_sigma_ms) * dt_s

                if use_pbl_cap and "hpbl" in m2 and np.isfinite(m2["hpbl"]):
                    hpbl = max(20.0, float(m2["hpbl"]))
                    z_new = clamp(z_new, 2.0, hpbl)
                else:
                    z_new = clamp(z_new, 2.0, 500.0)

                p.lat, p.lon, p.z_m = lat_new, lon_new, z_new

        centerlines.append({"z0": float(z0), "track": track_center})

    return centerlines, cloud_points


# ---------------------------
# GeoJSON writers
# ---------------------------

def centerlines_geojson(centerlines):
    feats = []
    for c in centerlines:
        coords = [[lon, lat] for (t, lat, lon, z) in c["track"]]
        feats.append({
            "type": "Feature",
            "properties": {"z0_m": c["z0"]},
            "geometry": {"type": "LineString", "coordinates": coords}
        })
    return {"type": "FeatureCollection", "features": feats}

def cloud_geojson(cloud_points, every_n: int = 1):
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


# ---------------------------
# Minimal test runner
# ---------------------------

if __name__ == "__main__":
    met = MetStoreV2("met_data")
    print("Loaded snapshots:", len(met.snaps))
    print("Time range:", met.snaps[0].valid, "to", met.snaps[-1].valid)

    # example start (change these)
    start_lat = 53.5461
    start_lon = -113.4938
    start_time = met.snaps[0].valid  # for a quick test, use first valid

    centers, cloud = run_back_trajectories(
        met,
        start_lat=start_lat,
        start_lon=start_lon,
        start_time_utc=start_time,
        hours=5.0,
        dt_s=60,
        n_particles=150
    )

    os.makedirs("odour_data", exist_ok=True)
    
    with open("odour_data/backtraj_centerlines.geojson", "w") as f:
        json.dump("odour_data/backtraj_centerlines_geojson", f)
    
    with open("odour_data/backtraj_cloud.geojson", "w") as f:
        json.dump("odour_data/backtraj_cloud_geojson", f)


    print("Wrote backtraj_centerlines.geojson and backtraj_cloud.geojson")
