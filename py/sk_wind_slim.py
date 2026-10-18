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
#   Supabase storage, public bucket "winds" (plain JSON: the storage CDN gzips it in transit):
#     sk_slim/gen2/<generation>/wind_<valid>.json   one folder per run of this script
#     sk_slim/index.json                            the current generation's times + files
#
# Station adjustment (wind_obs_nudge.py): every hour from now-6 h to the latest hour with
# ECCC station data is the model (linear in time between 3-hourly snapshots) corrected to
# that hour's station winds; later (forecast) times carry the latest correction, fading
# with lead time. With no station data at all the plain model is published.
#
# Runs hourly (run_sk_wind_slim.sh, cron) and after each ab_met_pull.py. Safe to re-run:
# files already converted are skipped.

import datetime as dt
import glob
import gzip
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wind_obs_nudge import buddy_check, correction_at, fetch_swob, sample_grid

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
# Bump when the crop/block/format changes: it names the published folder (sk_slim/gen<N>/)
# and published files are cached for a day (v1 = first, narrower crop on 2026-10-18).
SLIM_VERSION = 2

LOOKBACK_H = 6        # index starts this far before now (matches AB_winds/index.html)
CONVERT_BACK_H = 72   # only convert full files valid within this window (the georef check needs
                      # f000 hours ECCC's hourly archive has caught up on - it can lag >24 h)
LOCAL_KEEP_DAYS = 7
GEN_KEEP_H = 3        # older published generations are deleted after this

# Station adjustment - values from validate_sk_nudge.py (2026-10-18, 65 h, ~200 stations/h,
# held-out SK stations, with data/wind_station_exclude.csv + the hourly buddy check):
# vector RMSE 2.30 -> 1.96 m/s (-15%), direction within 45 deg 97.1 -> 98.5 %.
NUDGE_RADIUS_KM = 75   # best of 25-200 km
NUDGE_FADE_H = 6       # forecast hours keep exp(-lead/6 h) of the last correction (never fading got worse than raw by +3 h)
MIN_STATIONS = 30
ALERT_LOG = Path("/opt/airquality/logs/sitrep_alerts.log")

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
        if now.timestamp() - path.stat().st_mtime < 120:
            continue    # ab_met_pull.py may still be writing it (it writes in place) - next run gets it
        try:
            payload = make_slim(path)
        except (EOFError, OSError, ValueError, KeyError) as e:
            print(f"  skipped {path.name}: {type(e).__name__}: {e}")
            continue
        tmp = out.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), allow_nan=False)
        tmp.replace(out)
        made += 1
    print(f"converted {made} file(s)")

    # 2. model snapshots: newest run for every valid time from now-LOOKBACK_H-3h on
    best = {}
    for path in OUT_DIR.glob("sk_wind_*.json.gz"):
        m = re.search(r"sk_wind_(\d{8})_(\d{2})Z_f(\d{3})", path.name)
        run = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=UTC)
        lead = int(m.group(3))
        valid = run + dt.timedelta(hours=lead)
        if valid < now - dt.timedelta(hours=LOOKBACK_H + 3):
            continue
        if valid not in best or run > best[valid][0]:
            best[valid] = (run, lead, path)
    if not best:
        raise SystemExit("no slim wind files cover the index window - nothing to publish")
    snaps = {}
    for valid, (run, lead, path) in best.items():
        d = json.load(gzip.open(path, "rt"))
        g = d["grid"]
        arr = lambda a: np.array([np.nan if x is None else x for x in a], float).reshape(g["ny"], g["nx"])
        snaps[valid] = {"run": run, "lead": lead, "grid": g, "u": arr(d["u"]), "v": arr(d["v"])}
    grid = next(iter(snaps.values()))["grid"]
    glat = (grid["la1"] - grid["dy"] * np.arange(grid["ny"]))[:, None] * np.ones((1, grid["nx"]))
    glon = (grid["lo1"] + grid["dx"] * np.arange(grid["nx"]))[None, :] * np.ones((grid["ny"], 1))

    def model_at(t):
        """Model field at time t, linear in time between the 3-hourly snapshots."""
        vt = sorted(snaps)
        a = max((x for x in vt if x <= t), default=None)
        b = min((x for x in vt if x >= t), default=None)
        if a is None or b is None:
            return None
        if a == b:
            return snaps[a]["u"], snaps[a]["v"], snaps[a]
        w = (t - a) / (b - a)
        src = snaps[a] if w < 0.5 else snaps[b]
        return (snaps[a]["u"] * (1 - w) + snaps[b]["u"] * w,
                snaps[a]["v"] * (1 - w) + snaps[b]["v"] * w, src)

    # 3. station adjustment (wind_obs_nudge.py; R and fade from validate_sk_nudge.py)
    timeline = []   # (valid, u, v, src snapshot, adj-info)
    hour0 = now.replace(minute=0, second=0, microsecond=0)
    corr_last, last_hour = None, None
    for k in range(LOOKBACK_H, -1, -1):
        t = hour0 - dt.timedelta(hours=k)
        mod = model_at(t)
        if mod is None:
            continue
        try:
            obs = fetch_swob(t, (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX))
        except Exception as e:
            print(f"  {t:%m-%d %HZ}: station data unavailable ({e})")
            continue
        if len(obs) < MIN_STATIONS:
            continue
        mu, mv, src = mod
        slat = np.array([o["lat"] for o in obs]); slon = np.array([o["lon"] for o in obs])
        su, sv = sample_grid(grid, mu, mv, slat, slon)
        ok = np.isfinite(su) & np.isfinite(sv)
        ru = np.array([o["u"] for o in obs])[ok] - su[ok]
        rv = np.array([o["v"] for o in obs])[ok] - sv[ok]
        keep = buddy_check(slat[ok], slon[ok], ru, rv, NUDGE_RADIUS_KM)
        cu, cv = correction_at(slat[ok][keep], slon[ok][keep], ru[keep], rv[keep], glat, glon, NUDGE_RADIUS_KM)
        timeline.append((t, mu + cu, mv + cv, src,
                         {"kind": "stations", "n": int(keep.sum()), "dropped": int((~keep).sum())}))
        corr_last, last_hour = (cu, cv), t
    for valid in sorted(snaps):
        if last_hour is not None and valid <= last_hour:
            continue
        if last_hour is None and valid < now - dt.timedelta(hours=LOOKBACK_H):
            continue
        sn = snaps[valid]
        if corr_last is None:
            timeline.append((valid, sn["u"], sn["v"], sn, {"kind": "none"}))
        else:
            f = math.exp(-(valid - last_hour).total_seconds() / 3600 / NUDGE_FADE_H)
            timeline.append((valid, sn["u"] + f * corr_last[0], sn["v"] + f * corr_last[1], sn,
                             {"kind": "fading", "weight": round(f, 2),
                              "from": last_hour.isoformat().replace("+00:00", "Z")}))
    # alert only when station adjustment stops / restarts (state file), not every hour
    state_f = OUT_DIR / "adjust_state.json"
    was_ok = json.loads(state_f.read_text()).get("ok", True) if state_f.exists() else True
    if last_hour is None:
        print("WARNING: no station data for any recent hour - publishing the unadjusted model")
        if was_ok:
            with open(ALERT_LOG, "a") as f:
                f.write(f"{dt.datetime.now(UTC).isoformat(timespec='seconds')} ALERT sk_wind_slim: no ECCC station "
                        f"winds for the last {LOOKBACK_H} h - SK_Air_Map animated winds are the plain model (unadjusted)\n")
    else:
        print(f"adjusted to stations through {last_hour:%m-%d %HZ}")
        if not was_ok:
            with open(ALERT_LOG, "a") as f:
                f.write(f"{dt.datetime.now(UTC).isoformat(timespec='seconds')} ALERT sk_wind_slim: resolved - "
                        f"station adjustment back (through {last_hour:%m-%d %HZ})\n")
    state_f.write_text(json.dumps({"ok": last_hour is not None}))

    # 4. upload this generation's files into their own folder, then the index
    #    (the index never points at a missing file, and a page never mixes generations)
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    store = sb.storage.from_(BUCKET)
    gen = now.strftime("%Y%m%d%H%M")
    iso = lambda t: t.isoformat().replace("+00:00", "Z")
    enc = lambda a: [None if not np.isfinite(x) else round(float(x), 1) for x in a.ravel()]
    entries = []
    for valid, u, v, src, adj in timeline:
        name = f"{PREFIX}/gen{SLIM_VERSION}/{gen}/wind_{valid:%Y%m%d%H}.json"
        payload = {"meta": {"valid": iso(valid), "run": iso(src["run"]), "lead": src["lead"],
                            "grid_version": 2, "slim_version": SLIM_VERSION, "adjust": adj,
                            "winds": "10 m, earth-relative (u east, v north), m/s"},
                   "grid": grid, "u": enc(u), "v": enc(v)}
        store.upload(name, json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), {
            "content-type": "application/json", "cache-control": "86400", "upsert": "true"})
        entries.append({"t": iso(valid), "run": iso(src["run"]), "lead": src["lead"], "path": name, "adjust": adj})
    index = {"generated": iso(now.replace(microsecond=0)),
             "source": "HRDPS 10 m wind, adjusted to ECCC SWOB station winds (AB_winds/py/sk_wind_slim.py)",
             "adjust": {"radius_km": NUDGE_RADIUS_KM, "fade_h": NUDGE_FADE_H},
             "entries": entries}
    store.upload(f"{PREFIX}/index.json", json.dumps(index, separators=(",", ":")).encode(), {
        "content-type": "application/json", "cache-control": "60", "upsert": "true"})
    print(f"published generation {gen}: {len(entries)} times ({entries[0]['t']} -> {entries[-1]['t']})")

    # 5. cleanup: older generations (kept GEN_KEEP_H for pages still on an old index),
    #    pre-adjustment root files, and local slim files after LOCAL_KEEP_DAYS
    removed = 0
    for it in store.list(f"{PREFIX}/gen{SLIM_VERSION}", {"limit": 1000}):
        g = it["name"]
        if g != gen and re.fullmatch(r"\d{12}", g) and \
                dt.datetime.strptime(g, "%Y%m%d%H%M").replace(tzinfo=UTC) < now - dt.timedelta(hours=GEN_KEEP_H):
            files = [f"{PREFIX}/gen{SLIM_VERSION}/{g}/{f['name']}" for f in store.list(f"{PREFIX}/gen{SLIM_VERSION}/{g}", {"limit": 1000})]
            if files:
                store.remove(files)
                removed += len(files)
    old_root = [f"{PREFIX}/{it['name']}" for it in store.list(PREFIX, {"limit": 1000})
                if it["name"].startswith("sk_wind_")]
    if old_root:
        store.remove(old_root)
        removed += len(old_root)
    cutoff = now.timestamp() - LOCAL_KEEP_DAYS * 86400
    gone = 0
    for p in OUT_DIR.glob("sk_wind_*.json.gz"):
        if p.stat().st_mtime < cutoff:
            p.unlink()
            gone += 1
    print(f"removed {removed} remote / {gone} local old file(s)")


if __name__ == "__main__":
    sys.exit(main())
