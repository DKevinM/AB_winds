"""Known-release backward-LS test: Flesch, Wilson, Harper, Crenna & Sharpe (2004), "Deducing ground-to-air
emissions from observed trace gas concentrations: a field trial", J. Appl. Meteor. 43, 487-502.
Ellerslie (Alberta) 2001: methane from a 6 m x 6 m surface area source at a known rate Q; line-averaged
concentration C_L on open-path laser paths 17-201 m long, 0.8-1.0 m up, up to ~100 m downwind.

Data: flesch2004/table1.csv, transcribed from the paper's Table 1 (108 fifteen-minute periods, 13 trials;
the PDF's maths-font minus and times glyphs were mapped by font, not guessed).

Method (the paper's): for each period, release particles backward from points along the laser path,
count WindTrax touchdowns inside the source, (C/Q)_sim = (1/N) sum 2/|w0| [s/m]; then
Q_bLS = (C_L - C_b) / (C/Q)_sim and compare with the true Q. Met = the period's sonic u*, L, z0 and wind
direction (MOST profiles, our model's own turbulence formulas - not the measured sigmas).

Paper's own bLS result (their screening: drop trial A4 and |L| <= 2 m, n = 77): mean Q_bLS/Q = 1.02,
sd 0.36; sd 0.22 (n = 46) when the laser path was not at the plume edge.

Run from AB_winds/odour:  python validation/validate_flesch2004.py [points_per_path] [particles_per_point]
Writes validation/flesch2004/results.csv.
"""
import csv
import datetime as dt
import math
import multiprocessing as mp
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ls_model as ls          # noqa: E402

LAT0, LON0 = 53.415, -113.55                      # Ellerslie research farm (local frame origin)
M_LAT = 111320.0
M_LON = 111320.0 * math.cos(math.radians(LAT0))
SOURCE = [(-0.5, 0.5), (5.5, 0.7), (5.7, -5.3), (-0.3, -5.5)]   # paper Table 1 caption, (x east, y north) m
P_PA, M_CH4, R = 93000.0, 16.04, 8.314            # paper: assumed 930 hPa


def in_poly(x, y, poly=SOURCE):
    inside = np.zeros_like(x, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        inside ^= cond
    return inside


def ppm_to_gm3(ppm, T_C):
    return ppm * 1e-6 * P_PA * M_CH4 / (R * (T_C + 273.15))


def simulate(args):
    row, k_pts, n_per = args
    L = float(row["L"])
    met = ls.HomogeneousMet(ustar=float(row["ustar"]), L=L, zi=1000.0 if L < 0 else 500.0,
                            z0=float(row["z0"]), wind_to_deg=(float(row["beta"]) + 180.0) % 360.0)
    x1, y1, x2, y2, z = (float(row[k]) for k in ("lx1", "ly1", "lx2", "ly2", "lz"))
    corners = np.array(SOURCE)
    dmax = max(math.hypot(px - cx, py - cy) for px, py in ((x1, y1), (x2, y2)) for cx, cy in corners)
    U = max(float(row["U"]), 0.3)
    hours = min(1800.0, max(180.0, 2.0 * (dmax + 10.0) / U)) / 3600.0
    cfg = ls.LSConfig(dt_adv=5.0, z_reflect=max(float(row["z0"]), 0.02), record_every=10 ** 9,
                      dt_min=0.05, radius_km=None)
    t0 = dt.datetime(2001, 6, 1, tzinfo=dt.timezone.utc)
    cq = []
    for j, f in enumerate((np.arange(k_pts) + 0.5) / k_pts):
        px, py = x1 + f * (x2 - x1), y1 + f * (y2 - y1)
        cfg.seed = zlib.crc32(f'{row["trial"]}|{row["time"]}|{j}'.encode())
        res = ls.run_ls(met, LAT0 + py / M_LAT, LON0 + px / M_LON, t0, hours=hours, n_particles=n_per,
                        start_heights_m=(z,), direction="backward", cfg=cfg)
        if res.touchdowns:
            td = np.array([(la, lo, w0) for (_t, la, lo, w0, _z) in res.touchdowns])
            tx, ty = (td[:, 1] - LON0) * M_LON, (td[:, 0] - LAT0) * M_LAT
            ins = in_poly(tx, ty)
            cq.append(float(np.sum(2.0 / np.maximum(np.abs(td[ins, 2]), 1e-3)) / n_per))
        else:
            cq.append(0.0)
    cq_line = float(np.mean(cq))                          # line average of point C/Q, s/m
    dC = ppm_to_gm3(float(row["CL"]) - float(row["Cb"]), float(row["T_C"]))
    q_bls = dC / cq_line if cq_line > 0 else float("nan")
    return {**row, "cq_sim_s_per_m": cq_line, "dC_gm3": dC, "Q_bls": q_bls, "ratio": q_bls / float(row["Q"]),
            "sim_s": round(hours * 3600)}


def main(k_pts=15, n_per=1500):
    rows = list(csv.DictReader(open(os.path.join(HERE, "flesch2004", "table1.csv"))))
    with mp.get_context("fork").Pool(max(1, min(4, (os.cpu_count() or 2) - 1))) as pool:
        out = pool.map(simulate, [(r, k_pts, n_per) for r in rows])
    with open(os.path.join(HERE, "flesch2004", "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    summarise(out)


def summarise(out):
    def stats(sel, label):
        r = np.array([o["ratio"] for o in sel if math.isfinite(o["ratio"])])
        if r.size:
            print(f"  {label:44s} n={r.size:3d}  mean Q_bLS/Q {r.mean():.2f}  sd {r.std(ddof=1):.2f}  "
                  f"median {np.median(r):.2f}  within x2 {np.mean((r >= .5) & (r <= 2)):.0%}")
    keep = [o for o in out if not o["trial"].startswith("A4") and abs(float(o["L"])) > 2.0]
    print("Flesch et al. (2004) known-release test, AB_winds LS model:")
    stats(out, "all 108 periods")
    stats([o for o in out if not o["trial"].startswith("A4")], "excluding trial A4 (paper n=88: 1.27, sd 2.07)")
    stats(keep, "paper screening, no A4, |L|>2 m (paper n=77: 1.02, sd 0.36)")
    stats([o for o in keep if o["trial"].endswith("-5")], "  laser 5 (near the source)")
    stats([o for o in keep if o["trial"].endswith("-6")], "  laser 6 (farther, 60-140 m)")
    stats([o for o in keep if float(o["L"]) < 0], "  unstable (L<0)")
    stats([o for o in keep if float(o["L"]) > 0], "  stable (L>0)")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
