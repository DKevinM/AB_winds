# sk_wind_slim.py
#
# Small 10 m wind files for SK_Air_Map's animated wind layer.
#
# The full HRDPS files ab_met_pull.py writes are ~19 MB each (15 fields, 2.5 km,
# BC coast to Ontario) - far too heavy for a phone; SK_Air_Map already had a
# browser-crash episode with FireSmoke vectors. This crops ugrd10/vgrd10 to
# Saskatchewan plus a panning margin and block-averages 3x3 cells (~7 km),
# which is ~40 KB per time step.
#
# Georeferencing: each output value is the mean of a 3x3 block, placed at the
# block's CENTRE cell, so readers keep the usual convention
#   i = (lon - lo1) / dx ,  j = (la1 - lat) / dy
# on the output grid. scripts/check_georeference.py (sk_wind_slim) tests the
# result against ECCC station winds daily.
#
# Output:
#   local   met_data_sk/sk_wind_<run>_f<lead>.json.gz   (kept 7 days; the georef check reads these)
#   Supabase storage, public bucket "winds":
#     sk_slim/sk_wind_<run>_f<lead>.json      immutable, one per valid time in the index
#                                             (plain JSON: the storage CDN gzips it in transit)
#     sk_slim/index.json                      which file to use for each valid time (newest run wins)
#
# Runs after ab_met_pull.py in run_ab_met_pull.sh. Safe to re-run: files
# already converted are skipped, uploads are upserts.

import datetime as dt
import glob
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from supabase import create_client

ROOT = Path(__file__).resolve().parent.parent
MET_DIR = ROOT / "met_data"
OUT_DIR = ROOT / "met_data_sk"
BUCKET = "winds"
PREFIX = "sk_slim"

# SK_Air_Map bounds are [[48.9,-110.1],[60.1,-101.3]]; the wide margin is what a
# desktop screen shows at the default zoom 6 (Kamloops to Winnipeg), so the
# particles don't stop in a hard box just past the border.
LAT_MIN, LAT_MAX = 47.5, 61.5
LON_MIN, LON_MAX = -118.0, -94.0
BLOCK = 3
# Bump when the crop/block changes: remote files are immutable + cached for a day,
# so a new layout needs new URLs (v1 = first, narrower crop on 2026-10-18).
SLIM_VERSION = 2

LOOKBACK_H = 6        # index starts this far before now (matches AB_winds/index.html)
CONVERT_BACK_H = 72   # only convert full files valid within this window (the georef check needs
                      # f000 hours ECCC's hourly archive has caught up on - it can lag >24 h)
LOCAL_KEEP_DAYS = 7
REMOTE_KEEP_H = 48    # remote files not in the index are deleted after this

NAME_RE = re.compile(r"ab_met_(\d{8})_(\d{2})Z_f(\d{3})\.json\.gz$")
UTC = dt.timezone.utc


def parse_name(path):
    m = NAME_RE.search(str(path))
    if not m:
        return None
    run = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=UTC)
    lead = int(m.group(3))
    return run, lead, run + dt.timedelta(hours=lead)


def slim_name(run, lead):
    return f"sk_wind_{run:%Y%m%d_%HZ}_f{lead:03d}.json.gz"


def remote_name(local_path):
    return local_path.name[:-len(".gz")].replace("sk_wind_", f"sk_wind_v{SLIM_VERSION}_", 1)


def block_mean(a, j0, i0, ny, nx):
    """Mean of BLOCKxBLOCK blocks starting at (j0, i0); NaN only where a whole block is NaN."""
    sub = a[j0:j0 + ny * BLOCK, i0:i0 + nx * BLOCK].reshape(ny, BLOCK, nx, BLOCK)
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(sub).sum(axis=(1, 3))
        tot = np.nansum(sub, axis=(1, 3))
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def make_slim(full_path):
    with gzip.open(full_path, "rt") as f:
        d = json.load(f)
    if not d.get("meta", {}).get("grid_version"):
        raise ValueError(f"{full_path.name}: old-layout wind file (no grid_version) - refusing")
    g = d["grid"]
    U = np.array(d["fields"]["ugrd10"][0], dtype=float)
    V = np.array(d["fields"]["vgrd10"][0], dtype=float)
    assert U.shape == (g["ny"], g["nx"]) == V.shape, f"unexpected field shape {U.shape}"

    # full-grid index range covering the box (j counts south from la1)
    i0 = max(0, int(np.floor((LON_MIN - g["lo1"]) / g["dx"])))
    i1 = min(g["nx"], int(np.ceil((LON_MAX - g["lo1"]) / g["dx"])) + 1)
    j0 = max(0, int(np.floor((g["la1"] - LAT_MAX) / g["dy"])))
    j1 = min(g["ny"], int(np.ceil((g["la1"] - LAT_MIN) / g["dy"])) + 1)
    nx, ny = (i1 - i0) // BLOCK, (j1 - j0) // BLOCK

    u, v = block_mean(U, j0, i0, ny, nx), block_mean(V, j0, i0, ny, nx)
    grid = {
        # centre cell of the first block
        "lo1": round(g["lo1"] + (i0 + BLOCK // 2) * g["dx"], 6),
        "la1": round(g["la1"] - (j0 + BLOCK // 2) * g["dy"], 6),
        "dx": round(g["dx"] * BLOCK, 6),
        "dy": round(g["dy"] * BLOCK, 6),
        "nx": nx, "ny": ny,
    }
    enc = lambda a: [None if not np.isfinite(x) else round(float(x), 1) for x in a.ravel()]
    return {
        "meta": {**{k: d["meta"][k] for k in ("run", "lead", "valid", "grid_version")},
                 "slim_version": SLIM_VERSION,
                 "winds": "10 m, earth-relative (u east, v north), m/s",
                 "source": full_path.name,
                 "block": BLOCK},
        "grid": grid,
        "u": enc(u),
        "v": enc(v),
    }


def main():
    now = dt.datetime.now(UTC)
    OUT_DIR.mkdir(exist_ok=True)

    # 1. convert recent full files that don't have a slim twin yet
    made = 0
    for path in sorted(MET_DIR.glob("ab_met_*.json.gz")):
        info = parse_name(path)
        if not info or info[2] < now - dt.timedelta(hours=CONVERT_BACK_H):
            continue
        out = OUT_DIR / slim_name(info[0], info[1])
        if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
            continue
        payload = make_slim(path)
        tmp = out.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), allow_nan=False)
        tmp.replace(out)
        made += 1
    print(f"converted {made} file(s)")

    # 2. index: newest run for every valid time from now-LOOKBACK_H on
    best = {}
    for path in OUT_DIR.glob("sk_wind_*.json.gz"):
        m = re.search(r"sk_wind_(\d{8})_(\d{2})Z_f(\d{3})", path.name)
        run = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=UTC)
        lead = int(m.group(3))
        valid = run + dt.timedelta(hours=lead)
        if valid < now - dt.timedelta(hours=LOOKBACK_H):
            continue
        if valid not in best or run > best[valid][0]:
            best[valid] = (run, lead, path)
    if not best:
        raise SystemExit("no slim wind files cover the index window - nothing to publish")
    entries = [{"t": v.isoformat().replace("+00:00", "Z"), "run": r.isoformat().replace("+00:00", "Z"),
                "lead": l, "path": f"{PREFIX}/{remote_name(p)}"} for v, (r, l, p) in sorted(best.items())]

    # 3. upload referenced files, then the index (so the index never points at a missing file)
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    store = sb.storage.from_(BUCKET)
    existing = {it["name"] for it in store.list(PREFIX, {"limit": 1000})}
    up = 0
    for v, (r, l, p) in best.items():
        if remote_name(p) in existing:
            continue
        with gzip.open(p, "rb") as f:
            store.upload(f"{PREFIX}/{remote_name(p)}", f.read(), {
                "content-type": "application/json", "cache-control": "86400", "upsert": "true"})
        up += 1
    index = {"generated": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
             "source": "HRDPS 10 m wind via AB_winds/py/sk_wind_slim.py",
             "entries": entries}
    store.upload(f"{PREFIX}/index.json", json.dumps(index, separators=(",", ":")).encode(), {
        "content-type": "application/json", "cache-control": "60", "upsert": "true"})
    print(f"uploaded {up} file(s); index has {len(entries)} times "
          f"({entries[0]['t']} -> {entries[-1]['t']})")

    # 4. cleanup: remote files no longer indexed and older than REMOTE_KEEP_H; local after LOCAL_KEEP_DAYS
    keep = {remote_name(p) for (_, _, p) in best.values()} | {"index.json"}
    stale = []
    for name in existing - keep:
        m = re.search(r"sk_wind_(?:v\d+_)?(\d{8})_(\d{2})Z_f(\d{3})", name)
        if not m:
            continue
        valid = (dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=UTC)
                 + dt.timedelta(hours=int(m.group(3))))
        if valid < now - dt.timedelta(hours=REMOTE_KEEP_H):
            stale.append(f"{PREFIX}/{name}")
    if stale:
        store.remove(stale)
    cutoff = now.timestamp() - LOCAL_KEEP_DAYS * 86400
    gone = 0
    for p in OUT_DIR.glob("sk_wind_*.json.gz"):
        if p.stat().st_mtime < cutoff:
            p.unlink()
            gone += 1
    print(f"removed {len(stale)} remote / {gone} local old file(s)")


if __name__ == "__main__":
    sys.exit(main())
