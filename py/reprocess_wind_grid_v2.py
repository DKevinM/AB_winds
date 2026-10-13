# reprocess_wind_grid_v2.py
#
# One-off (2026-10-13): convert ab_met JSON files written before the
# hrdps_regrid.py fix into the grid_version 2 layout.
#
# Old files hold the raw ROTATED HRDPS array, cropped with
# ds.where(box, drop=True), rows south->north and u/v grid-relative, under a
# bounding-box header that every reader treated as a north-up regular grid.
# The HRDPS grid never changes, so the crop can be located exactly in the
# full grid and resampled/rotated just as ab_met_pull.py now does live.
#
#   python py/reprocess_wind_grid_v2.py local      # met_data/, originals moved to BACKUP_DIR
#   python py/reprocess_wind_grid_v2.py supabase   # overwrite every file in the 'winds' bucket
#
# Both are idempotent: files whose meta already has grid_version are skipped.

import argparse
import datetime as dt
import glob
import gzip
import json
import os
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
import requests
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hrdps_regrid import Regridder, GRID_VERSION  # noqa: E402

WIND_LEVELS = (10, 40, 80, 120)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET_DIR = os.path.join(ROOT, "met_data")
BACKUP_DIR = "/opt/airquality/backups/met_data_gridv1_20261013"
REF_URL = ("https://dd.weather.gc.ca/today/model_hrdps/continental/2.5km/00/000/"
           "{d}T00Z_MSC_HRDPS_UGRD_AGL-10m_RLatLon0.0225_PT000H.grib2")

# Crop boxes ab_met_pull.py used over its life (git history).
OLD_BOXES = [
    (48, 62, -125, -88),    # ace4874, 2026-08-12 onward
    (48, 62, -125, -100),   # c2bec79, 2026-07-28 .. 2026-08-12
]

_RG = None
_LAYOUTS = None  # header tuple -> (j0, i0, ny, nx)


def _old_header(la, lo, box):
    """Exactly what the old crop()+header code produced for this box."""
    la_min, la_max, lo_min, lo_max = box
    lon360 = lo % 360
    m = (la >= la_min) & (la <= la_max) & (lon360 >= lo_min % 360) & (lon360 <= lo_max % 360)
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    j0, j1, i0, i1 = rows[0], rows[-1], cols[0], cols[-1]
    assert len(rows) == j1 - j0 + 1 and len(cols) == i1 - i0 + 1, "crop not contiguous"
    sla = la[j0:j1 + 1, i0:i1 + 1]
    slo = lo[j0:j1 + 1, i0:i1 + 1]
    slo = np.where(slo > 180, slo - 360, slo)
    ny, nx = sla.shape
    lat_max = float(np.nanmax(sla))
    lon_min = float(np.nanmin(slo))
    dx = float((np.nanmax(slo) - np.nanmin(slo)) / (nx - 1))
    dy = float((lat_max - float(np.nanmin(sla))) / (ny - 1))
    return (round(lon_min, 6), round(lat_max, 6), round(dx, 12), round(dy, 12), nx, ny), (int(j0), int(i0), ny, nx)


def _hdr_key(g):
    return (round(float(g["lo1"]), 6), round(float(g["la1"]), 6),
            round(float(g["dx"]), 12), round(float(g["dy"]), 12), int(g["nx"]), int(g["ny"]))


def init(ref_grib):
    global _RG, _LAYOUTS
    _RG = Regridder(ref_grib)
    ds = xr.open_dataset(ref_grib, engine="cfgrib")
    la, lo = ds.latitude.values, ds.longitude.values
    ds.close()
    _LAYOUTS = dict(_old_header(la, lo, b) for b in OLD_BOXES)


def convert(obj):
    """Old-layout payload dict -> grid_version 2 payload dict (raises if unrecognised)."""
    layout = _LAYOUTS.get(_hdr_key(obj["grid"]))
    if layout is None:
        raise ValueError(f"unknown grid header {obj['grid']}")
    j0, i0, ny, nx = layout
    full = np.full((_RG.src.nj, _RG.src.ni), np.nan)
    out = {}
    for k, v in obj["fields"].items():
        a = np.array(v[0] if len(v) == 1 else v, dtype=np.float64)   # None -> nan
        if a.shape != (ny, nx):
            raise ValueError(f"{k}: shape {a.shape} != header {(ny, nx)}")
        full[:] = np.nan
        full[j0:j0 + ny, i0:i0 + nx] = a
        out[k] = _RG.resample(full)
    for h in WIND_LEVELS:
        uk, vk = f"ugrd{h}", f"vgrd{h}"
        if uk in out and vk in out:
            out[uk], out[vk] = _RG.earth_relative(out[uk], out[vk])
        else:
            out.pop(uk, None)
            out.pop(vk, None)
    meta = dict(obj["meta"])
    meta["grid_version"] = GRID_VERSION
    meta["winds"] = "earth-relative (u east, v north), m/s"
    meta["reprocessed"] = "2026-10-13 from grid_version 1 (misplaced rotated layout)"
    return {"meta": meta, "grid": _RG.grid,
            "fields": {k: [np.round(a.astype(np.float64), 2).tolist()] for k, a in out.items()}}


def dumps_gz(payload, path):
    # NaN -> null: JSON.parse in the browser rejects NaN tokens
    txt = json.dumps(payload, allow_nan=True).replace("NaN", "null")
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:   # level 9 is ~8x slower here
        f.write(txt)
    os.replace(tmp, path)


def _is_v2(obj):
    return "grid_version" in obj.get("meta", {})


def do_local(path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            obj = json.load(f)
        if _is_v2(obj):
            return ("skip", path)
        new = convert(obj)
        bak = os.path.join(BACKUP_DIR, os.path.basename(path))
        if not os.path.exists(bak):
            os.link(path, bak)          # same filesystem: keeps the original inode, no copy
        dumps_gz(new, path)             # atomic replace of the live path
        return ("ok", path)
    except Exception as e:
        return ("error", f"{path}: {e}")


def _sb():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return url, {"apikey": key, "Authorization": f"Bearer {key}"}


def do_supabase(file_path):
    url, h = _sb()
    try:
        r = requests.get(f"{url}/storage/v1/object/winds/{file_path}", headers=h, timeout=120)
        r.raise_for_status()
        stored_gzip = r.content[:2] == b"\x1f\x8b"
        remote = json.loads((gzip.decompress(r.content) if stored_gzip else r.content).decode("utf-8"))
        if _is_v2(remote) and stored_gzip:
            return ("skip", file_path)
        if _is_v2(remote):
            # converted earlier but stored uncompressed (2026-10-13 upload bug): re-gzip, re-upload
            payload = remote
        else:
            payload = None
        # back up the original unless the identical bytes are already in the local backup
        local_bak = os.path.join(BACKUP_DIR, os.path.basename(file_path))
        same = os.path.exists(local_bak) and open(local_bak, "rb").read() == r.content
        if payload is None and not same:
            bak = os.path.join(BACKUP_DIR, "supabase", file_path.replace("/", "__"))
            if not os.path.exists(bak):
                with open(bak, "wb") as f:
                    f.write(r.content)
        fd, tmp = tempfile.mkstemp(suffix=".json.gz")
        os.close(fd)
        dumps_gz(payload if payload is not None else convert(remote), tmp)
        with open(tmp, "rb") as f:
            body = f.read()
        os.remove(tmp)
        up = requests.put(f"{url}/storage/v1/object/winds/{file_path}",
                          # gzip bytes stored as-is, like the originals. Do NOT send
                          # Content-Encoding: gzip - the gateway inflates the body and the
                          # object ends up stored as plain JSON, which every reader rejects.
                          headers={**h, "Content-Type": "application/json", "x-upsert": "true"},
                          data=body, timeout=300)
        if up.status_code >= 300:
            return ("error", f"{file_path}: upload {up.status_code} {up.text[:200]}")
        return ("ok", file_path)
    except Exception as e:
        return ("error", f"{file_path}: {e}")


def _ref_grib():
    d = dt.datetime.utcnow().strftime("%Y%m%d")
    p = os.path.join(tempfile.gettempdir(), f"hrdps_ref_{d}.grib2")
    if not os.path.exists(p):
        r = requests.get(REF_URL.format(d=d), timeout=120)
        r.raise_for_status()
        with open(p, "wb") as f:
            f.write(r.content)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["local", "supabase"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.join(BACKUP_DIR, "supabase"), exist_ok=True)
    ref = _ref_grib()

    if args.target == "local":
        items = sorted(glob.glob(os.path.join(MET_DIR, "ab_met_*.json.gz")), reverse=True)  # newest first
        fn = do_local
    else:
        url, h = _sb()
        items, off = [], 0
        while True:
            r = requests.get(f"{url}/rest/v1/wind_files?select=file_path&order=valid_time.desc",
                             headers={**h, "Range": f"{off}-{off + 999}"}, timeout=60)
            r.raise_for_status()
            rows = r.json()
            items += [x["file_path"] for x in rows]
            if len(rows) < 1000:
                break
            off += 1000
        fn = do_supabase
    if args.limit:
        items = items[:args.limit]

    print(f"{len(items)} {args.target} files, {args.workers} workers", flush=True)
    counts = {"ok": 0, "skip": 0, "error": 0}
    with Pool(args.workers, initializer=init, initargs=(ref,)) as pool:
        for n, (status, msg) in enumerate(pool.imap_unordered(fn, items), 1):
            counts[status] += 1
            if status == "error":
                print("ERROR", msg, flush=True)
            if n % 50 == 0 or n == len(items):
                print(f"{n}/{len(items)} {counts}", flush=True)
    print("done", counts, flush=True)


if __name__ == "__main__":
    main()
