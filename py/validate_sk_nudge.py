# validate_sk_nudge.py
#
# Leave-one-station-out check of the SK wind station adjustment (wind_obs_nudge.py),
# same idea as the Hinton / Howe Sound validate.py: for every past hour, each station
# is predicted from the model corrected by all the OTHER stations, and scored against
# what it measured. Picks the correction radius R, then the fade time for forecasts.
#
#   python3 py/validate_sk_nudge.py [hours_back=66]
#
# Model = the raw slim SK files in met_data_sk/ (newest run per valid time, linear in
# time between the 3-hourly valid times) - exactly what the map shows unadjusted.
# Scored at stations inside SK; stations in the margin still help correct.

import datetime as dt
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wind_obs_nudge import buddy_check, excluded_names, fetch_swob, gc_km, sample_grid

ROOT = Path(__file__).resolve().parent.parent
SLIM = ROOT / "met_data_sk"
CACHE = ROOT / "met_data_sk" / "obs_cache"
BBOX = (-118.0, 47.5, -94.0, 61.5)
SK = (49.0, 60.0, -110.0, -101.36)       # scoring box (lat0, lat1, lon0, lon1)
UTC = dt.timezone.utc
RADII = [25, 50, 75, 100, 150, 200]
TAUS = [1, 3, 6, 12, 24, 1e9]             # hours; 1e9 = never fades
LEADS = [1, 2, 3, 6, 9, 12]


def load_snapshots():
    best = {}
    for p in SLIM.glob("sk_wind_*.json.gz"):
        m = re.search(r"sk_wind_(\d{8})_(\d{2})Z_f(\d{3})", p.name)
        run = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=UTC)
        valid = run + dt.timedelta(hours=int(m.group(3)))
        if valid not in best or run > best[valid][0]:
            best[valid] = (run, p)
    snaps = {}
    for valid, (run, p) in best.items():
        d = json.load(gzip.open(p, "rt"))
        g = d["grid"]
        arr = lambda a: np.array([np.nan if x is None else x for x in a], float).reshape(g["ny"], g["nx"])
        snaps[valid] = (g, arr(d["u"]), arr(d["v"]))
    return snaps


def model_at(snaps, t, lat, lon):
    vt = sorted(snaps)
    a = max((x for x in vt if x <= t), default=None)
    b = min((x for x in vt if x >= t), default=None)
    if a is None or b is None:
        return None
    ua, va = sample_grid(*snaps[a], lat, lon)
    if a == b:
        return ua, va
    ub, vb = sample_grid(*snaps[b], lat, lon)
    w = (t - a) / (b - a)
    return ua * (1 - w) + ub * w, va * (1 - w) + vb * w


def obs_at(t):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{t:%Y%m%d%H}.json"
    if f.exists():
        skip = excluded_names()
        return [o for o in json.loads(f.read_text()) if (o["name"] or "").strip().upper() not in skip]
    o = fetch_swob(t, BBOX)
    f.write_text(json.dumps(o))
    return o


def loo_corr(lat, lon, ru, rv, R):
    """Leave-one-out correction at every station, using only stations that pass the
    hourly buddy check as helpers (the held-out station is still scored either way)."""
    keep = buddy_check(lat, lon, ru, rv, R)
    D = gc_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    W = np.exp(-(D / R) ** 2) * keep[None, :]
    np.fill_diagonal(W, 0.0)
    den = np.maximum(W.sum(1), 1.0)
    return (W @ ru) / den, (W @ rv) / den


def score(u, v, ou, ov, ospd):
    """dict: direction-within-45 % (obs >= 10 km/h), vector RMSE m/s, speed bias m/s, n."""
    spd = np.hypot(u, v)
    strong = ospd >= 10 / 3.6
    ang = np.degrees(np.abs(np.angle((u + 1j * v) / (ou + 1j * ov + 1e-12))))
    return {"dir45": float(np.mean(ang[strong] <= 45) * 100) if strong.any() else np.nan,
            "rmse": float(np.sqrt(np.mean((u - ou) ** 2 + (v - ov) ** 2))),
            "bias": float(np.mean(spd - ospd)), "n": int(len(u))}


def pooled(rows):
    rows = [r for r in rows if r["n"]]
    n = sum(r["n"] for r in rows)
    agg = lambda k: sum(r[k] * r["n"] for r in rows if np.isfinite(r[k])) / max(1, sum(r["n"] for r in rows if np.isfinite(r[k])))
    return {"dir45": agg("dir45"), "rmse": agg("rmse"), "bias": agg("bias"), "n": n}


def main(hours_back=66):
    snaps = load_snapshots()
    now = dt.datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    hours = [now - dt.timedelta(hours=h) for h in range(2, hours_back + 1)]
    hours = [t for t in hours if model_at(snaps, t, np.array([55.0]), np.array([-106.0])) is not None]
    print(f"{len(hours)} hours with model coverage, {min(hours):%m-%d %HZ} .. {max(hours):%m-%d %HZ}")

    per_hour = {}
    for t in hours:
        o = obs_at(t)
        if len(o) < 30:
            continue
        lat = np.array([x["lat"] for x in o]); lon = np.array([x["lon"] for x in o])
        ou = np.array([x["u"] for x in o]); ov = np.array([x["v"] for x in o]); osp = np.array([x["spd"] for x in o])
        mu, mv = model_at(snaps, t, lat, lon)
        ok = np.isfinite(mu) & np.isfinite(mv)
        per_hour[t] = dict(lat=lat[ok], lon=lon[ok], ou=ou[ok], ov=ov[ok], osp=osp[ok], mu=mu[ok], mv=mv[ok],
                           ids=[x["id"] for x, k in zip(o, ok) if k])
    print(f"{len(per_hour)} hours with >= 30 stations; median stations/hour "
          f"{int(np.median([len(h['lat']) for h in per_hour.values()]))}")

    # 1. analysis hours: raw model vs LOO-adjusted, by radius
    raw, by_r = [], {R: [] for R in RADII}
    for t, h in per_hour.items():
        insk = (h["lat"] >= SK[0]) & (h["lat"] <= SK[1]) & (h["lon"] >= SK[2]) & (h["lon"] <= SK[3])
        raw.append(score(h["mu"][insk], h["mv"][insk], h["ou"][insk], h["ov"][insk], h["osp"][insk]))
        ru, rv = h["ou"] - h["mu"], h["ov"] - h["mv"]
        for R in RADII:
            cu, cv = loo_corr(h["lat"], h["lon"], ru, rv, R)
            by_r[R].append(score((h["mu"] + cu)[insk], (h["mv"] + cv)[insk], h["ou"][insk], h["ov"][insk], h["osp"][insk]))
    fmt = lambda s: f"dir within 45: {s['dir45']:5.1f}%   vector RMSE {s['rmse']:.2f} m/s   speed bias {s['bias']:+.2f} m/s   (n={s['n']})"
    print("\nANALYSIS HOURS (held-out SK stations)")
    print(f"  raw HRDPS        {fmt(pooled(raw))}")
    for R in RADII:
        print(f"  adjusted R={R:3d}  {fmt(pooled(by_r[R]))}")
    best_r = min(RADII, key=lambda R: pooled(by_r[R])["rmse"])
    print(f"  -> best radius by vector RMSE: {best_r} km")

    # 2. forecast fade: residuals at hour t (LOO), applied to t+k with weight exp(-k/tau)
    print(f"\nFORECAST FADE (R={best_r} km; correction from hour t applied at t+k, held-out station)")
    for k in LEADS:
        rows_raw, rows_tau = [], {tau: [] for tau in TAUS}
        for t, h in per_hour.items():
            t2 = t + dt.timedelta(hours=k)
            if t2 not in per_hour:
                continue
            h2 = per_hour[t2]
            common = sorted(set(h["ids"]) & set(h2["ids"]))
            if len(common) < 30:
                continue
            i1 = np.array([h["ids"].index(s) for s in common]); i2 = np.array([h2["ids"].index(s) for s in common])
            lat, lon = h["lat"][i1], h["lon"][i1]
            ru, rv = (h["ou"] - h["mu"])[i1], (h["ov"] - h["mv"])[i1]
            cu, cv = loo_corr(lat, lon, ru, rv, best_r)
            insk = (lat >= SK[0]) & (lat <= SK[1]) & (lon >= SK[2]) & (lon <= SK[3])
            mu2, mv2 = h2["mu"][i2][insk], h2["mv"][i2][insk]
            ou2, ov2, os2 = h2["ou"][i2][insk], h2["ov"][i2][insk], h2["osp"][i2][insk]
            rows_raw.append(score(mu2, mv2, ou2, ov2, os2))
            for tau in TAUS:
                f = np.exp(-k / tau)
                rows_tau[tau].append(score(mu2 + f * cu[insk], mv2 + f * cv[insk], ou2, ov2, os2))
        if not rows_raw:
            continue
        line = "  ".join(f"tau {('never' if tau > 1e8 else str(tau)+'h'):>5}: {pooled(rows_tau[tau])['rmse']:.2f}" for tau in TAUS)
        print(f"  +{k:2d} h  raw {pooled(rows_raw)['rmse']:.2f}   {line}   (vector RMSE m/s)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 66)
