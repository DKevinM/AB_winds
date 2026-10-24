"""Backfill HRDPS from ECCC's HPFX mirror (~50 days kept) in the normal ab_met format.

dd.weather.gc.ca only keeps ~1 day, so ab_met_pull can't go back. HPFX
(hpfx.collab.science.gc.ca/<YYYYMMDD>/WXO-DD/...) keeps about 50 days. This
rebuilds files with the CURRENT grid (grid_version 2, west edge -134 since
2026-10-14, so the BC coast is included) into a staging folder:

  * runs 00/06/12/18Z, leads 0-5 by default (hourly coverage, newest run per hour)
  * same fields, regrid and earth-relative rotation as ab_met_pull.process_run
  * NO Supabase upload (it keeps 30 days and is fed by the live pull)

hpfx_merge.py then checks each staged file against any existing one and moves
it into met_data/, after which the normal wind archive applies (92 days here,
everything in Dropbox). 2026-10-24, Kevin: "pull the 50 days, keep that history".

Usage: hpfx_backfill.py START_DATE END_DATE [--leads 0,1,2,3,4,5] [--workers 3]
"""

import argparse
import datetime as dt
import gzip
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import requests
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hrdps_regrid import regridder_for, GRID_VERSION   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.path.join(ROOT, "met_data_hpfx")
HPFX = "https://hpfx.collab.science.gc.ca/{d:%Y%m%d}/WXO-DD/model_hrdps/continental/2.5km"
# identical to ab_met_pull.FIELDS / LEVELS_AGL (kept here so importing ab_met_pull's Supabase client isn't needed)
LEVELS_AGL = [10, 40, 80, 120]
FIELDS = {"UGRD": [f"AGL-{h}m" for h in LEVELS_AGL], "VGRD": [f"AGL-{h}m" for h in LEVELS_AGL],
          "TMP": [f"AGL-{h}m" for h in (2, 40, 80, 120)], "HPBL": ["Sfc"], "RH": ["AGL-2m"], "DPT": ["AGL-2m"]}


def url(run, var, level, lead):
    return (f"{HPFX.format(d=run)}/{run:%H}/{lead:03d}/{run:%Y%m%dT%HZ}_MSC_HRDPS_{var}_{level}"
            f"_RLatLon0.0225_PT{lead:03d}H.grib2")


def fetch(u):
    for attempt in range(4):
        try:
            r = requests.get(u, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            f = tempfile.NamedTemporaryFile(delete=False, suffix=".grib2")
            f.write(r.content); f.close()
            return f.name
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
    return None


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def do_lead(run, lead):
    fname = os.path.join(STAGE, f"ab_met_{run:%Y%m%d_%HZ}_f{lead:03d}.json.gz")
    if os.path.exists(fname):
        return "exists"
    fields, grid, missing = {}, None, []
    for var, levels in FIELDS.items():
        for lvl in levels:
            tmp = fetch(url(run, var, lvl, lead))
            if tmp is None:
                missing.append(f"{var}_{lvl}"); continue
            try:
                rg = regridder_for(tmp)
                ds = xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""})
                arr = ds.to_array().values[0]; ds.close()
                grid = rg.grid
                fields[f"{var.lower()}{lvl.replace('AGL-', '').replace('m', '')}"] = rg.resample(arr)
            finally:
                os.remove(tmp)
    if not fields or any(k not in fields for k in ("ugrd10", "vgrd10")):
        return f"skipped (missing {', '.join(missing[:4])})"
    for h in LEVELS_AGL:
        uk, vk = f"ugrd{h}", f"vgrd{h}"
        if uk in fields and vk in fields:
            fields[uk], fields[vk] = rg.earth_relative(fields[uk], fields[vk])
        else:
            fields.pop(uk, None); fields.pop(vk, None)
    fields = {k: [np.round(a.astype(np.float64), 2).tolist()] for k, a in fields.items()}
    payload = {"meta": {"run": run.isoformat() + "Z", "lead": lead, "valid": (run + dt.timedelta(hours=lead)).isoformat() + "Z",
                        "grid_version": GRID_VERSION, "winds": "earth-relative (u east, v north), m/s",
                        "source": "HPFX backfill (hpfx_backfill.py)"},
               "grid": grid, "fields": fields}
    part = fname + ".part"
    with gzip.open(part, "wt", encoding="utf-8") as f:
        json.dump(clean(payload), f, allow_nan=False)
    os.replace(part, fname)
    return "ok" + (f" (missing {len(missing)})" if missing else "")


def do_run(args):
    run, leads = args
    out = []
    for lead in leads:
        try:
            out.append((lead, do_lead(run, lead)))
        except Exception as ex:
            out.append((lead, f"error {type(ex).__name__}: {ex}"))
    return run, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start"); ap.add_argument("end")
    ap.add_argument("--leads", default="0,1,2,3,4,5")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(STAGE, exist_ok=True)
    d0, d1 = dt.datetime.fromisoformat(a.start), dt.datetime.fromisoformat(a.end)
    leads = [int(x) for x in a.leads.split(",")]
    runs = []
    d = d0
    while d <= d1:
        runs += [d.replace(hour=h) for h in (0, 6, 12, 18)]
        d += dt.timedelta(days=1)
    print(f"{len(runs)} runs x {len(leads)} leads -> {STAGE}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for run, res in ex.map(do_run, [(r, leads) for r in runs]):
            print(f"{dt.datetime.utcnow():%H:%M:%S} {run:%Y-%m-%d %HZ}: " + ", ".join(f"f{l:03d} {s}" for l, s in res), flush=True)


if __name__ == "__main__":
    main()
