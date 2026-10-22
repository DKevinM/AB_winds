"""Project Sagebrush (NOAA ARL Field Research Division, Idaho National Laboratory) - forward LS check.

PSB1 (Oct 2013, NOAA TM OAR ARL-268): 5 daytime tests, near-neutral to unstable, arcs 200-3200 m.
PSB2 (2016, NOAA TM OAR ARL-275): tests 1-4 summer daytime; 5-7 pre-dawn and 8 evening - very stable,
light-wind nights (2 m wind ~1 m/s, u* 0.03-0.1 m/s, L a few metres), arcs 100-800 m.
SF6 released continuously at 1.5 m agl from the grid centre; 10-min bag samples at 1 m agl on arcs.
Data: /opt/airquality/data/validation/sagebrush/{2013,2016} (original-format files from NOAA ARL).

Measured: for each 10-min bag period and arc, crosswind-integrated Cy/Q = sum(C * r dtheta) / Q along the
arc (trapezoid over the samplers' angles), C = SF6 above background. Background = 7 pptv (reports: ambient
6-8 pptv); flags 1-2 (below MLOQ) count as zero excess; flags 3-5 (missing) are dropped. Arcs whose plume
runs off the array (an end sampler >= 10% of the arc maximum) or that barely saw the plume (max excess
< 20 pptv) are excluded, as NOAA did for its sigma-y work.

Model: forward LS (same crossing-plane estimator as validate_prairie_grass.py) in a homogeneous surface
layer driven by that period's sonic u*, L and scalar wind speed at the sonic height (PSB1: G1 4 m beside
the release; PSB2: R2 2 m on the GRI tower, R9 3.7 m as fallback); z0 = 0.035 m (reports' median 3-3.8
cm); BL depth 1500 m by day, 0.4 sqrt(u* L / f) at night (Zilitinkevich), clipped to 30-800 m.
Release 1.5 m, receptor band 0.5-1.5 m (bags at 1 m).

Run from AB_winds/odour:  python validation/validate_sagebrush.py [n_particles]
Writes validation/sagebrush/sagebrush_results.csv.
"""
import csv
import datetime as dt
import glob
import math
import multiprocessing as mp
import os
import sys
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import ls_model as ls                                        # noqa: E402
from validate_prairie_grass import scores                    # noqa: E402

DATA = "/opt/airquality/data/validation/sagebrush"
Z_SRC, Z_RCP, DZ, Z0 = 1.5, 1.0, 1.0, 0.035
BKG_PPT = 7.0
P_PA = 85000.0                         # Grid 3, INL, ~1500 m elevation
M_SF6, R_GAS = 146.06, 8.314
F_COR = 2 * 7.292e-5 * math.sin(math.radians(43.59))
# measured 2-h release rates, g/s (report Table 2 of each phase)
Q = {("PSB1", 1): 10.177, ("PSB1", 2): 9.986, ("PSB1", 3): 9.930, ("PSB1", 4): 1.043, ("PSB1", 5): 1.030,
     ("PSB2", 1): 0.1922, ("PSB2", 2): 0.1460, ("PSB2", 3): 0.1218, ("PSB2", 4): 0.1466,
     ("PSB2", 5): 0.0147, ("PSB2", 6): 0.0120, ("PSB2", 7): 0.0120, ("PSB2", 8): 0.0119}


_trapz = getattr(np, "trapezoid", None) or np.trapz


def ppt_to_gm3(ppt, T_C):
    return ppt * 1e-12 * P_PA * M_SF6 / (R_GAS * (T_C + 273.15))


def read_sonic(path, z):
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    d["t"] = pd.to_datetime(d["file"].str[2:12], format="%m%d%y%H%M")     # LLMMDDYYHHMM, period start, MST
    d = d.replace(-9999.0, np.nan)
    return d.set_index("t")[["SCALWS", "VECWD", "USTR_rot", "OLEN_rot", "TAVG"]].assign(z=z)


def sonics():
    """{(phase, iop): [primary DataFrame, fallback DataFrame or None]}"""
    out = {}
    g1 = read_sonic(f"{DATA}/2013/sagebrush2013/meteorology/arlfrd sonics/gri tower/G1_PSB1data_10min_October.csv", 4.0)
    for iop in range(1, 6):
        out[("PSB1", iop)] = [g1, None]
    for iop in range(1, 9):
        base = f"{DATA}/2016/sagebrush2016/Meteorology/Sonic/FRD/IOP{iop}/PSB2data_FRD_IOP{iop}"
        out[("PSB2", iop)] = [read_sonic(f"{base}_R2_2m_10min.csv", 2.0), read_sonic(f"{base}_R9_3.7m_10min.csv", 3.7)]
    return out


def bags():
    files = [("PSB1", f) for f in sorted(glob.glob(f"{DATA}/2013/sagebrush2013/bagsampling/PSB1_IOP*_BagSampling_Final.csv"))]
    files += [("PSB2", f) for f in sorted(glob.glob(f"{DATA}/2016/sagebrush2016/BagSampling/Sage16Sampler0*.CSV"))]
    frames = []
    for phase, f in files:
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        d = d[(d["begin day"] > 0) & (d["agl"] == 1)]
        d["phase"] = phase
        frames.append(d)
    d = pd.concat(frames)
    d["t"] = pd.to_datetime(d["begin day"].astype(int).astype(str) + d["beg. tm"].astype(int).astype(str).str.zfill(6),
                            format="%Y%m%d%H%M%S")
    return d


def measured_periods():
    b, son = bags(), sonics()
    rows = []
    for (phase, iop, t, dist), g in b.groupby(["phase", "Test", "t", "dist"]):
        g = g[~g.flag.isin([3, 4, 5])].sort_values("angle")
        if len(g) < 5:
            continue
        exc = np.where(g.flag.isin([1, 2]), 0.0, np.maximum(g.ppt.values - BKG_PPT, 0.0))
        mx = exc.max()
        trunc = exc[0] >= 0.1 * mx or exc[-1] >= 0.1 * mx
        # met for this period: primary sonic, else fallback
        met = None
        for s in son[(phase, iop)]:
            if s is not None and t in s.index and np.isfinite(s.loc[t, ["SCALWS", "USTR_rot", "OLEN_rot", "TAVG"]].astype(float)).all():
                met = s.loc[t]
                break
        if met is None:
            continue
        T = float(met.TAVG)
        th = np.radians(g.angle.values.astype(float))
        cy = float(_trapz(ppt_to_gm3(exc, T), th * dist))           # g/m2 along the arc
        rows.append({"phase": phase, "iop": int(iop), "t": t.isoformat(), "dist": int(dist), "n_samplers": len(g),
                     "max_excess_ppt": round(float(mx), 1), "truncated": bool(trunc), "Q": Q[(phase, int(iop))],
                     "cyq_obs": cy / Q[(phase, int(iop))], "U": float(met.SCALWS), "z_son": float(met.z),
                     "ustar": float(met.USTR_rot), "L": float(met.OLEN_rot), "T_C": T, "wd": float(met.VECWD)})
    return pd.DataFrame(rows)


def simulate(key):
    phase, iop, t, ustar, L, U, z_son, dists = key
    if L > 0:
        zi = float(np.clip(0.4 * math.sqrt(max(ustar, 0.01) * L / F_COR), 30.0, 800.0))
    else:
        zi = 1500.0
    met = ls.HomogeneousMet(ustar=max(ustar, 0.01), L=L, zi=zi, z0=Z0, wind_to_deg=90.0,
                            U_ref=max(U, 0.2), z_ref=z_son)
    hours = min(3.0, (1.3 * max(dists) / max(U, 0.3)) * 3.0 / 3600.0)
    cfg = ls.LSConfig(dt_adv=10.0, z_reflect=Z0 * 5, record_every=10 ** 9, dt_min=0.05, crossing_both=CROSS_BOTH,
                      seed=zlib.crc32(f"{phase}|{iop}|{t}".encode()))
    res = ls.run_ls(met, 43.59066, -112.938, dt.datetime(2016, 1, 1, tzinfo=dt.timezone.utc), hours=hours,
                    n_particles=N_PART, start_heights_m=(Z_SRC,), direction="forward", cfg=cfg,
                    crossing_planes_m=dists, crossing_dz=DZ, crossing_z=Z_RCP)
    return key, {d: res.crossings[float(d)] for d in dists}, zi


N_PART = 3000
CROSS_BOTH = False   # True: count plane crossings in both directions (proper estimator in light wind)


def main(n_particles=3000):
    global N_PART
    N_PART = n_particles
    m = measured_periods()
    keys = [(p, i, t, g.ustar.iloc[0], g.L.iloc[0], g.U.iloc[0], g.z_son.iloc[0], tuple(sorted(g.dist.unique())))
            for (p, i, t), g in m.groupby(["phase", "iop", "t"])]
    with mp.get_context("fork").Pool(max(1, min(4, (os.cpu_count() or 2) - 1))) as pool:
        sims = pool.map(simulate, keys)
    pred = {}
    zis = {}
    for (p, i, t, *_rest), cq, zi in sims:
        zis[(p, i, t)] = zi
        for d, v in cq.items():
            pred[(p, i, t, d)] = v
    m["cyq_ls"] = [pred.get((r.phase, r.iop, r.t, r.dist), np.nan) for r in m.itertuples()]
    m["zi"] = [zis.get((r.phase, r.iop, r.t)) for r in m.itertuples()]
    m.to_csv(os.path.join(HERE, "sagebrush", "sagebrush_results.csv"), index=False)
    summarise(m)


def summarise(m):
    ok = m[(~m.truncated) & (m.max_excess_ppt >= 20)]
    night = lambda x: x[(x.phase == "PSB2") & (x.iop >= 5)]
    day = lambda x: x[~((x.phase == "PSB2") & (x.iop >= 5))]
    print(f"Project Sagebrush, forward LS (Cy/Q at 1 m): {len(m)} arc-periods, {len(ok)} usable (not truncated, max >= 20 ppt above bkg)")
    for label, x in [("ALL usable", ok), ("PSB1 2013 (day)", ok[ok.phase == "PSB1"]), ("PSB2 day (tests 1-4)", ok[(ok.phase == "PSB2") & (ok.iop <= 4)]),
                     ("PSB2 NIGHT (tests 5-8)", night(ok)), ("  night, L < 10 m (very stable)", night(ok)[night(ok).L < 10])]:
        if len(x) < 3:
            print(f"  {label:34s} n={len(x)}")
            continue
        s = scores(list(x.cyq_obs), list(x.cyq_ls))
        print(f"  {label:34s} n={s['n']:3d}  FB={s['FB']:+.2f}  NMSE={s['NMSE']:.2f}  FAC2={s['FAC2']:.2f}  MG={s['MG']:.2f}")
    print("  by arc (usable):")
    for d, x in ok.groupby("dist"):
        if len(x) >= 3:
            s = scores(list(x.cyq_obs), list(x.cyq_ls))
            print(f"    {d:5d} m  n={s['n']:3d}  FB={s['FB']:+.2f}  FAC2={s['FAC2']:.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
