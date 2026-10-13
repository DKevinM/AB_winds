"""
Validate the LS and Gaussian models against Project Prairie Grass (1956) -
the same dataset Kevin's STLS workbook used (Lagrangian Model_b.xlsm,
"Project Prairie Grass Data" sheet, extracted to prairie_grass.csv).

Setup (Barad 1958; standard for this dataset): SO2 released at 0.46 m,
samplers at 1.5 m on arcs 50-800 m downwind, z0 = 0.006 m. Observed
crosswind-integrated Cy/Q (s/m2) is compared with:
  * LS forward model, driven by each run's measured u*, L and 2 m wind;
  * Gaussian plume, Briggs rural sigma_z for the run's PG class (the
    textbook screening approach);
  * Gaussian with continuous similarity spread (gaussian_puff's growth
    law, the default in the puff model), transport at the effective height.

Scores (Chang & Hanna 2004): FB fractional bias, NMSE, FAC2 (fraction
within a factor of 2), geometric mean bias MG.

Run from AB_winds/odour:  python validation/validate_prairie_grass.py [n_particles]
"""

import csv
import datetime as dt
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import ls_model as ls          # noqa: E402
import gaussian_plume as gp    # noqa: E402
import gaussian_puff as gpf    # noqa: E402
import micromet as mm          # noqa: E402

ARCS = (50, 100, 200, 400, 800)
Z_SRC = 0.46
Z_RCP = 1.5
Z0 = 0.006


def load_runs():
    runs = []
    with open(os.path.join(HERE, "prairie_grass.csv")) as f:
        for r in csv.DictReader(f):
            try:
                run = {"exp": int(float(r["exp"])), "U2": float(r["U2m"]), "ustar": float(r["ustar"]),
                       "L": float(r["L"]), "pg": r["pg"].strip()}
            except (ValueError, KeyError):
                continue
            zi = r.get("zi", "")
            run["zi"] = float(zi) if zi not in ("", None) and float(zi) < 4999 else 1000.0
            run["obs"] = {}
            for x in ARCS:
                v = r.get(f"CyQ_{x}", "")
                if v not in ("", None):
                    run["obs"][x] = float(v)
            if run["obs"]:
                runs.append(run)
    return runs


def scores(obs, mod):
    o = np.asarray(obs)
    m = np.asarray(mod)
    ok = (o > 0) & (m > 0)
    o, m = o[ok], m[ok]
    fb = 2 * (o.mean() - m.mean()) / (o.mean() + m.mean())
    nmse = np.mean((o - m) ** 2) / (o.mean() * m.mean())
    fac2 = np.mean((m / o >= 0.5) & (m / o <= 2.0))
    mg = math.exp(np.mean(np.log(o)) - np.mean(np.log(m)))
    return {"n": int(ok.sum()), "FB": fb, "NMSE": nmse, "FAC2": fac2, "MG": mg}


def main(n_particles=2000):
    runs = load_runs()
    t0 = dt.datetime(1956, 7, 15, 18, tzinfo=dt.timezone.utc)
    rows = []
    for run in runs:
        met = ls.HomogeneousMet(ustar=run["ustar"], L=run["L"], zi=run["zi"], z0=Z0,
                                wind_to_deg=90.0, U_ref=run["U2"], z_ref=2.0)
        U_mean = max(run["U2"], 0.5)
        hours = (900.0 / U_mean) * 3.0 / 3600.0      # time to clear 800 m with margin
        cfg = ls.LSConfig(dt_adv=10.0, z_reflect=Z0 * 5, record_every=10 ** 9, dt_min=0.05, seed=run["exp"])
        res = ls.run_ls(met, 41.0, -98.0, t0, hours=hours, n_particles=n_particles,
                        start_heights_m=(Z_SRC,), direction="forward", cfg=cfg,
                        crossing_planes_m=ARCS, crossing_dz=2.0, crossing_z=Z_RCP)
        wst = (-run["ustar"] ** 3 * run["zi"] / (0.4 * run["L"])) ** (1 / 3) if run["L"] < 0 else 0.0
        for x, obs in run["obs"].items():
            g = gp.crosswind_integrated(x, z=Z_RCP, H=Z_SRC, u=run["U2"], pg=run["pg"], zi=run["zi"])
            U = run["U2"]
            for _ in range(3):      # sigma_z and transport speed depend on each other
                sz = gpf.similarity_sigma_z(x, U, Z_SRC, run["ustar"], run["L"], run["zi"], wst)
                U = run["U2"] * float(mm.wind_profile_ratio(max(Z_SRC + gpf.ZE_FACTOR * sz, 0.5), 2.0, Z0, run["L"]))
            gs = float(gp._vertical_term(Z_RCP, Z_SRC, sz, run["zi"])) / U
            rows.append({"exp": run["exp"], "pg": run["pg"], "L": run["L"], "x": x, "obs": obs,
                         "ls": res.crossings[float(x)], "gauss": g, "gauss_sim": gs})
        print(f"run {run['exp']:>3} PG {run['pg']} L={run['L']:>7}: " + "  ".join(
            f"{x}m obs {run['obs'].get(x, float('nan')):.4f} ls {res.crossings[float(x)]:.4f}"
            for x in ARCS if x in run["obs"]), flush=True)

    with open(os.path.join(HERE, "prairie_grass_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nOverall (all arcs, all runs)")
    for name in ("ls", "gauss", "gauss_sim"):
        s = scores([r["obs"] for r in rows], [r[name] for r in rows])
        print(f"  {name:9s} n={s['n']:3d}  FB={s['FB']:+.2f}  NMSE={s['NMSE']:.2f}  FAC2={s['FAC2']:.2f}  MG={s['MG']:.2f}")
    print("\nBy stability")
    for grp, test in (("unstable (L<0)", lambda r: r["L"] < 0), ("stable (L>0)", lambda r: r["L"] > 0)):
        sub = [r for r in rows if test(r)]
        for name in ("ls", "gauss", "gauss_sim"):
            s = scores([r["obs"] for r in sub], [r[name] for r in sub])
            print(f"  {grp:15s} {name:9s} n={s['n']:3d}  FB={s['FB']:+.2f}  FAC2={s['FAC2']:.2f}  MG={s['MG']:.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
