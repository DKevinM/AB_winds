"""Project Prairie Grass, BACKWARD mode (2026-10-21).

The forward check (validate_prairie_grass.py) releases at the source and counts crossings of the
sampler arcs. This one runs the same model the other way, the way the Gold Bar inversion and DSAI
use it: release at the sampler (1.5 m), run backward in time, and estimate the crosswind-integrated
Cy/Q at each arc distance two ways:

  crossing   - crossings of the plane x metres upwind within a band around the real 0.46 m release
               height, weight 1/(N u dz). Exact Prairie Grass geometry; mirror of the forward estimator.
  touchdown  - the WindTrax bLS estimator Gold Bar uses: treat the release as a ground-level strip
               source of width 2*dx at x upwind; Cy/Q = sum(2/|w0|) / (N * 2*dx) over touchdowns in
               the strip. Approximates the 0.46 m release as ground level.

Homogeneous turbulence means one backward release covers all five arcs at once.
Run from AB_winds/odour:  python validation/validate_prairie_grass_backward.py [n_particles]
Writes validation/prairie_grass_backward_results.csv.
"""
import csv
import datetime as dt
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import ls_model as ls                                               # noqa: E402
from validate_prairie_grass import load_runs, scores, ARCS, Z_SRC, Z_RCP, Z0   # noqa: E402

DZ_SRC = 0.8            # crossing band 0.06-0.86 m around the 0.46 m release height
STRIP_FRAC = 0.1        # touchdown strip half-width = 10% of the arc distance


def main(n_particles=2000):
    runs = load_runs()
    t0 = dt.datetime(1956, 7, 15, 18, tzinfo=dt.timezone.utc)
    lat0, lon0 = 41.0, -98.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    rows = []
    for run in runs:
        met = ls.HomogeneousMet(ustar=run["ustar"], L=run["L"], zi=run["zi"], z0=Z0,
                                wind_to_deg=90.0, U_ref=run["U2"], z_ref=2.0)
        U_mean = max(run["U2"], 0.5)
        hours = (900.0 / U_mean) * 3.0 / 3600.0
        cfg = ls.LSConfig(dt_adv=10.0, z_reflect=Z0 * 5, record_every=10 ** 9, dt_min=0.05, seed=run["exp"])
        res = ls.run_ls(met, lat0, lon0, t0, hours=hours, n_particles=n_particles,
                        start_heights_m=(Z_RCP,), direction="backward", cfg=cfg,
                        crossing_planes_m=ARCS, crossing_dz=DZ_SRC, crossing_z=Z_SRC)
        # touchdowns: x upwind of the receptor (wind blows toward +east, so upwind = west)
        td = res.touchdowns
        xu = np.array([(lon0 - lo) * m_per_deg_lon for (_t, _la, lo, _w, _z) in td]) if td else np.array([])
        w = np.array([2.0 / max(abs(w0), 1e-3) for (_t, _la, _lo, w0, _z) in td]) if td else np.array([])
        for x, obs in run["obs"].items():
            dx = STRIP_FRAC * x
            sel = (xu >= x - dx) & (xu <= x + dx)
            td_cq = float(w[sel].sum() / (n_particles * 2 * dx)) if sel.any() else 0.0
            rows.append({"exp": run["exp"], "pg": run["pg"], "L": run["L"], "x": x, "obs": obs,
                         "ls_back_cross": res.crossings[float(x)], "ls_back_td": td_cq})
        print(f"run {run['exp']:>3} PG {run['pg']} L={run['L']:>7}: " + "  ".join(
            f"{x}m obs {run['obs'][x]:.4f} cross {res.crossings[float(x)]:.4f}"
            for x in ARCS if x in run["obs"]), flush=True)

    with open(os.path.join(HERE, "prairie_grass_backward_results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("\nOverall (all arcs, all runs)")
    for name in ("ls_back_cross", "ls_back_td"):
        s = scores([r["obs"] for r in rows], [r[name] for r in rows])
        print(f"  {name:14s} n={s['n']:3d}  FB={s['FB']:+.2f}  NMSE={s['NMSE']:.2f}  FAC2={s['FAC2']:.2f}  MG={s['MG']:.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
