import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import floor, isfinite
from typing import List, Optional, Tuple


# ----------------------------
# Data models (same idea as before)
# ----------------------------
@dataclass(frozen=True)
class WindSample:
    time_utc: datetime
    u10: float  # m/s eastward
    v10: float  # m/s northward


# ----------------------------
# Grid utilities
# ----------------------------
def _parse_valid_time(header: dict) -> datetime:
    """
    HRDPS earth-style JSON usually has:
      header.refTime (ISO string)
      header.forecastTime (hours)
    valid_time = refTime + forecastTime hours
    """
    ref = header.get("refTime")
    fh = header.get("forecastTime", 0)
    if ref is None:
        raise ValueError("header.refTime missing")

    # Example: "2025-12-29T12:00:00.000Z"
    ref_dt = datetime.fromisoformat(ref.replace("Z", "+00:00")).astimezone(timezone.utc)
    return ref_dt + timedelta(hours=float(fh))


def _grid_ij_float(lat: float, lon: float, header: dict) -> Tuple[float, float]:
    """
    Convert lat/lon to fractional i,j indices.

    Assumptions (typical for these JSON winds):
      lon(i) = lo1 + i*dx
      lat(j) = la1 - j*dy   (la1 is the northern edge)
    """
    lo1 = float(header["lo1"])
    la1 = float(header["la1"])
    dx = float(header["dx"])
    dy = float(header["dy"])

    i_f = (lon - lo1) / dx
    j_f = (la1 - lat) / dy
    return i_f, j_f


def _at(data: list, nx: int, ny: int, i: int, j: int) -> Optional[float]:
    """Fetch value at integer (i,j). Returns None if out of bounds or null."""
    if i < 0 or i >= nx or j < 0 or j >= ny:
        return None
    v = data[j * nx + i]
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    if not isfinite(fv):
        return None
    return fv


def bilinear_interp(lat: float, lon: float, header: dict, data: list) -> Optional[float]:
    """
    Bilinear interpolate grid value at (lat, lon).

    Returns None if insufficient valid neighbors. Falls back to nearest valid
    neighbor among the 4 corners when possible.
    """
    nx = int(header["nx"])
    ny = int(header["ny"])

    i_f, j_f = _grid_ij_float(lat, lon, header)

    # Outside grid?
    if i_f < 0 or j_f < 0 or i_f > (nx - 1) or j_f > (ny - 1):
        return None

    i0 = int(floor(i_f))
    j0 = int(floor(j_f))
    i1 = min(i0 + 1, nx - 1)
    j1 = min(j0 + 1, ny - 1)

    di = i_f - i0
    dj = j_f - j0

    q00 = _at(data, nx, ny, i0, j0)
    q10 = _at(data, nx, ny, i1, j0)
    q01 = _at(data, nx, ny, i0, j1)
    q11 = _at(data, nx, ny, i1, j1)

    # If all missing, nothing we can do
    corners = [(q00, (i0, j0)), (q10, (i1, j0)), (q01, (i0, j1)), (q11, (i1, j1))]
    valid = [(v, ij) for v, ij in corners if v is not None]
    if not valid:
        return None

    # If any corner missing, fall back to nearest valid corner (simple + robust)
    if any(v is None for v, _ in corners):
        # nearest in fractional space
        best = None
        best_d2 = 1e18
        for v, (ii, jj) in valid:
            d2 = (i_f - ii) ** 2 + (j_f - jj) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = v
        return float(best)

    # Bilinear blend
    return float(
        q00 * (1 - di) * (1 - dj)
        + q10 * di * (1 - dj)
        + q01 * (1 - di) * dj
        + q11 * di * dj
    )


# ----------------------------
# Read u/v JSON and sample at a point
# ----------------------------
def load_uv_at_point(u_path: str, v_path: str, lat: float, lon: float) -> WindSample:
    """
    Load a pair of JSON grids and interpolate u/v at the complaint point.
    """
    with open(u_path, "r", encoding="utf-8") as f:
        u_json = json.load(f)
    with open(v_path, "r", encoding="utf-8") as f:
        v_json = json.load(f)

    uh = u_json["header"]
    vh = v_json["header"]

    # Basic sanity checks
    for k in ("lo1", "la1", "dx", "dy", "nx", "ny"):
        if uh.get(k) != vh.get(k):
            raise ValueError(f"U/V grid mismatch on header key '{k}'")

    t_valid = _parse_valid_time(uh)

    u_val = bilinear_interp(lat, lon, uh, u_json["data"])
    v_val = bilinear_interp(lat, lon, vh, v_json["data"])

    if u_val is None or v_val is None:
        raise ValueError("Could not interpolate u/v at point (outside grid or null neighborhood).")

    return WindSample(time_utc=t_valid, u10=u_val, v10=v_val)


def build_hourly_samples_from_pairs(
    pairs: List[Tuple[str, str]],
    lat: float,
    lon: float
) -> List[WindSample]:
    """
    pairs = [(u_file_000, v_file_000), (u_file_001, v_file_001), ...]
    Returns WindSamples sorted by time_utc descending or ascending (your choice).
    """
    out: List[WindSample] = []
    for u_path, v_path in pairs:
        out.append(load_uv_at_point(u_path, v_path, lat, lon))
    out.sort(key=lambda s: s.time_utc, reverse=False)  # ascending
    return out
