import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import math

from .utils_time import floor_to_hour, ensure_utc

def _load_wind_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _bilinear(grid, lon, lat):
    """
    Minimal placeholder bilinear sampler.
    You must adapt this to match your AB_wind JSON structure.
    Return u,v in m/s at (lon,lat).
    """
    raise NotImplementedError("Hook this to your existing AB_wind grid format.")

class WindCache:
    def __init__(self, winds_dir: str):
        self.winds_dir = Path(winds_dir)
        self._mem = {}  # (slot)->json

    def load_slot(self, slot_filename: str) -> dict:
        if slot_filename not in self._mem:
            self._mem[slot_filename] = _load_wind_json(self.winds_dir / slot_filename)
        return self._mem[slot_filename]

class WindField:
    """
    Interpolates wind in time between two hourly fields.
    Assumes you can map a datetime to the correct cached json.
    """
    def __init__(self, cache: WindCache, slot_map_func):
        """
        slot_map_func(dt_hour)-> filename, e.g. returns "AB_wind_latest.json"
        or returns one of AB_wind_000..003 depending on which hour is requested.
        """
        self.cache = cache
        self.slot_map_func = slot_map_func

    def uv_at(self, lon: float, lat: float, t: datetime):
        t = ensure_utc(t)
        h0 = floor_to_hour(t)
        h1 = h0 - timedelta(hours=1)

        f = (t - h0).total_seconds() / 3600.0  # [0,1)

        j0 = self.cache.load_slot(self.slot_map_func(h0))
        j1 = self.cache.load_slot(self.slot_map_func(h1))

        u0, v0 = _bilinear(j0, lon, lat)
        u1, v1 = _bilinear(j1, lon, lat)

        u = (1 - f) * u0 + f * u1
        v = (1 - f) * v0 + f * v1
        return u, v
