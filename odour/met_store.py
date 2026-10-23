"""
Faster HRDPS loader for the new models.

backtraj_core.MetStoreV2 downloads every snapshot in +/- window_hours of
the start time from Supabase - including duplicate valid times from
overlapping forecast runs (32 files for a 10 h window) - and decodes each
~12-million-number JSON. A 6 h forward run spent >10 min just loading.

LocalMetStore (a MetStoreV2, so sample()/_bracket() and every caller work
unchanged):
  * reads the copies ab_met_pull already keeps in AB_winds/met_data,
    falling back to Supabase only for valid times missing locally;
  * keeps ONE snapshot per valid time - the newest run (shortest lead);
  * loads only [t_from, t_to] (the hours the run actually covers);
  * caches decoded fields as .npz (winds/PBL as float16; temperatures
    float32, since the 2-40 m difference that sets stability is often
    <1 K and float16 rounds to 0.125 K near 270 K), so a repeat run on
    the same day skips the JSON entirely.
"""

import datetime as dt
import glob
import os
import re
from typing import Optional

import sys

import numpy as np

import backtraj_core as bc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
import wind_archive  # noqa: E402  (older months live in Dropbox; restored on demand)

MET_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "met_data")
CACHE_DIR = "/opt/airquality/dsai_data/met_npz_cache"
_FN = re.compile(r"ab_met_(\d{8})_(\d{2})Z_f(\d{3})\.json\.gz$")


def _index(met_dir):
    """valid_time -> (run_time, lead, path), newest run per valid time."""
    best = {}
    for p in glob.glob(os.path.join(met_dir, "ab_met_*.json.gz")):
        m = _FN.search(p)
        if not m:
            continue
        run = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
        lead = int(m.group(3))
        valid = run + dt.timedelta(hours=lead)
        cur = best.get(valid)
        if cur is None or run > cur[0]:
            best[valid] = (run, lead, p)
    return best


class LocalMetStore(bc.MetStoreV2):
    def __init__(self, t_from: dt.datetime, t_to: dt.datetime, met_dir: str = MET_DIR_DEFAULT,
                 pad_hours: float = 3.0, supabase_fallback: bool = True, use_cache: bool = True):
        self.folder = met_dir
        self.bucket = "winds"
        self.model = "HRDPS"
        self.use_supabase = False
        self.cache = {}
        self.snaps = []
        t0 = min(t_from, t_to) - dt.timedelta(hours=pad_hours)
        t1 = max(t_from, t_to) + dt.timedelta(hours=pad_hours)
        if os.path.abspath(met_dir) == os.path.abspath(wind_archive.MET_DIR):
            utc = lambda t: t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
            try:
                wind_archive.ensure_window(utc(t0), utc(t1), met_dir, pad_hours=0)
            except Exception as ex:  # Dropbox down etc. - carry on with whatever is local
                print(f"wind_archive restore failed: {ex}", file=sys.stderr)
        idx = _index(met_dir)
        wanted = sorted(v for v in idx if t0 <= v <= t1)
        for v in wanted:
            self.snaps.append(self._load_cached(idx[v][2], use_cache))
        self.source = f"local met_data ({len(self.snaps)} snapshots)"
        if not self.snaps and supabase_fallback:
            # nothing local for this window (e.g. older than local retention)
            mid = t_from + (t_to - t_from) / 2
            span = abs((t_to - t_from).total_seconds()) / 3600.0 / 2 + pad_hours
            bc.MetStoreV2.__init__(self, folder=met_dir, use_supabase=True, target_time=mid,
                                   window_hours=int(np.ceil(span)))
            self.source = f"Supabase ({len(self.snaps)} snapshots)"
            return
        if not self.snaps:
            raise RuntimeError(f"No HRDPS snapshots between {t0} and {t1}")
        self.snaps.sort(key=lambda s: s.valid)
        self.grid = self.snaps[0].grid

    def _load_cached(self, path, use_cache):
        # ".g2": grid_version 2 files (regular north-up grid, earth-relative winds,
        # 2026-10-13). Caches built from the old misplaced layout are never reused.
        key = os.path.basename(path).replace(".json.gz", ".g2.npz")
        cpath = os.path.join(CACHE_DIR, key)
        if use_cache and os.path.exists(cpath):
            try:
                z = np.load(cpath, allow_pickle=False)
                g = z["grid"]
                grid = bc.Grid(lo1=float(g[0]), la1=float(g[1]), dx=float(g[2]), dy=float(g[3]),
                               nx=int(g[4]), ny=int(g[5]))
                meta = z["meta"]
                fields = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("f_")}
                return bc.MetSnapshot(valid=bc.parse_iso_z(str(meta[0])), run=bc.parse_iso_z(str(meta[1])),
                                      lead=int(meta[2]), grid=grid, fields=fields)
            except Exception:
                pass                              # corrupt cache -> rebuild below
        snap = self._load_one_local(path)
        if use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            g = snap.grid
            tmp = cpath + ".tmp.npz"
            np.savez_compressed(
                tmp, grid=np.array([g.lo1, g.la1, g.dx, g.dy, g.nx, g.ny], dtype=np.float64),
                meta=np.array([snap.valid.isoformat(), snap.run.isoformat(), str(snap.lead)]),
                **{"f_" + k: v.astype(np.float32 if k.startswith(("tmp", "dpt")) else np.float16)
                   for k, v in snap.fields.items()})
            os.replace(tmp, cpath)
        return snap
