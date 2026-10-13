"""
Side-by-side Prairie Grass comparison of:
  legacy  - backtraj_core.run_back_trajectories (current AB_winds model),
            forward mode, given the exact MOST log-profile wind at each
            particle height (its best case - no ASL/AGL height bug here)
  stls    - faithful numpy port of Kevin's STLModel VBA macro
            (Lagrangian Model_b.xlsm), bugs included, workbook settings
            (mu = 0.0025, pm_mean = 2.5, CDrag from Reynolds number),
            perfect reflection ON (the workbook had it OFF, which absorbs
            every particle that touches the ground)
  new LS  - ls_model.run_ls (taken from prairie_grass_results.csv)

legacy and stls are scored with the same residence-time estimator:
  Cy/Q(x) = sum(dt spent in [x +/- 0.1x] x [0.5, 2.5 m]) / (N * 0.2x * 2 m)

Writes model_comparison_pg.csv next to this file.
"""

import csv
import datetime as dt
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ODOUR = os.path.dirname(HERE)
sys.path.insert(0, ODOUR)

import micromet as mm                                   # noqa: E402
from validate_prairie_grass import load_runs, ARCS, scores, Z_SRC, Z0  # noqa: E402

ZBAND = (0.5, 2.5)


def residence_cyq(xs, zs, dts, n):
    out = {}
    for x in ARCS:
        dx = 0.2 * x
        m = (xs >= x - dx / 2) & (xs < x + dx / 2) & (zs >= ZBAND[0]) & (zs <= ZBAND[1])
        out[x] = float(np.sum(dts[m]) / (n * dx * (ZBAND[1] - ZBAND[0])))
    return out


# ---------------------------------------------------------------------------
# legacy model (exec'd with cloud recorded every step instead of every 5th)
# ---------------------------------------------------------------------------

def _legacy_module():
    src = open(os.path.join(ODOUR, "backtraj_core.py")).read()
    src = src.replace("if k % 5 == 0:", "if True:")
    mod = types.ModuleType("legacy_bt")
    mod.__dict__["__name__"] = "legacy_bt"
    exec(compile(src, "backtraj_core_legacy", "exec"), mod.__dict__)
    return mod


class _LegacyMet:
    """MetStoreV2 stand-in: exact log-profile wind blowing east."""
    def __init__(self, run):
        self.run = run

    def sample(self, t, lat, lon, z_m):
        r = self.run
        U = r["U2"] * float(mm.wind_profile_ratio(max(z_m, 0.01), 2.0, Z0, r["L"]))
        return {"u": U, "v": 0.0, "shear": 0.0, "hpbl": r["zi"]}


def run_legacy(run, n=300, dt_s=2):
    bt = _legacy_module()
    t0 = dt.datetime(1956, 7, 15, 18, tzinfo=dt.timezone.utc)
    hours = 900.0 / max(run["U2"], 0.5) * 2.0 / 3600.0
    _, cloud = bt.run_back_trajectories(
        _LegacyMet(run), 41.0, -98.0, t0, hours=hours, dt_s=dt_s, n_particles=n,
        start_heights_m=(Z_SRC,), direction="forward", use_terrain_steering=False)
    lat0, lon0 = 41.0, -98.0
    xs = np.array([math.radians(p[2] - lon0) * 6371000.0 * math.cos(math.radians(lat0)) for p in cloud])
    zs = np.array([p[3] for p in cloud])
    return residence_cyq(xs, zs, np.full(xs.size, float(dt_s)), n)


# ---------------------------------------------------------------------------
# Kevin's STLModel VBA, ported line-for-line (vectorized over particles)
# ---------------------------------------------------------------------------

def run_stls(run, n=500, mu=0.0025, pm_mean=2.5, kvis=1.761935e-05, seed=1, max_steps=400000):
    rng = np.random.default_rng(seed)
    sv, L, zo = run["ustar"], run["L"], Z0
    wind_vel = run["U2"]
    z_zero = Z_SRC
    x_end = 1000.0
    z_end = 250.0
    kv = 0.4
    if L < 0:
        sigw = 1.25 * sv * (1 - 3 * z_zero / L) ** (1 / 3)
        sigv = sv * (12 - 0.5 * 700 / L) ** (1 / 3)
        sigu = sigv
    else:  # VBA: L>0 stable, L=0 neutral - identical constants
        sigw, sigv, sigu = 1.3 * sv, 1.9 * sv, 2.4 * sv
    a = sv / kv
    b = (2 * sigw ** 2 * mu) ** 0.5
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.full(n, z_zero)
    w = sigw * rng.standard_normal(n)
    V = sigv * rng.standard_normal(n)
    U = sigu * rng.standard_normal(n)
    acc = {xa: 0.0 for xa in ARCS}
    alive = np.ones(n, dtype=bool)
    for _ in range(max_steps):
        if not alive.any():
            break
        zp = np.maximum(z, zo * 1.0001)
        if L < 0:
            TL = (0.75 * zp / sigw) * (1 - 6 * zp / L) ** 0.25
            d = (1 - 16 * zp / L) ** -0.25
            psi = 2 * np.log(0.5 * (1 + 1 / d)) + np.log(0.5 * (1 + 1 / d ** 2)) - 2 * np.arctan(1 / d) + np.pi / 2
            ux = a * (np.log(zp / zo) - psi)
        else:
            TL = (0.75 * zp / sigw) / (1 + 5 * zp / L)
            ux = a * (np.log(zp / zo) + 5 * zp / L)
        dtt = mu * TL
        du = -U * (sigw ** 2 / sigu ** 2) * mu + b * rng.standard_normal(n)
        Re = np.abs(du) * pm_mean / kvis
        Cd = np.select([Re == 0, Re <= 1, Re <= 400, Re <= 3e5, Re <= 2e6],
                       [0.0, 24 / np.maximum(Re, 1e-12), 24 / np.maximum(Re, 1e-12) ** 0.646, 0.5,
                        0.000366 * np.maximum(Re, 1e-12) ** 0.4275], 0.18)
        U = Cd * (U + du)
        x_new = x + wind_vel * dtt + (U + ux) * dtt          # mean wind added twice (as in VBA)
        dv = -V * (sigw ** 2 / sigv ** 2) * mu + b * rng.standard_normal(n)
        V = Cd * (V + dv)
        dw = -w * mu + b * rng.standard_normal(n)
        w = Cd * (w + dw)                                    # gas: vw = vd ~ 0
        z_new = z + w * dtt
        refl = z_new < zo
        w = np.where(refl, -w, w)
        z_new = np.where(refl, 2 * zo - z_new, z_new)
        done = (z_new > z_end) | (x_new >= x_end)
        inband = alive & (z >= ZBAND[0]) & (z <= ZBAND[1])
        if inband.any():
            for xa in ARCS:
                m = inband & (x >= 0.9 * xa) & (x < 1.1 * xa)
                if m.any():
                    acc[xa] += float(dtt[m].sum())
        alive_next = alive & ~done
        x = np.where(alive_next, x_new, 1e9)   # finished particles are frozen
        z = np.where(alive_next, z_new, z_end)
        alive = alive_next
    return {xa: acc[xa] / (n * 0.2 * xa * (ZBAND[1] - ZBAND[0])) for xa in ARCS}


def _stls100(run):
    return run_stls(run, n=100)


def main():
    runs = load_runs()
    newres = {}
    with open(os.path.join(HERE, "prairie_grass_results.csv")) as f:
        for r in csv.DictReader(f):
            newres[(int(r["exp"]), int(r["x"]))] = (float(r["ls"]), float(r["gauss"]))
    rows = []
    from multiprocessing import Pool
    with Pool(3) as pool:   # 3 of 4 cores - leave one for the production crons
        legs = pool.map(run_legacy, runs)
        stls = pool.map(_stls100, runs)
    for run, leg, stl in zip(runs, legs, stls):
        for x, obs in run["obs"].items():
            ls_v, g_v = newres.get((run["exp"], x), (float("nan"), float("nan")))
            rows.append({"exp": run["exp"], "pg": run["pg"], "L": run["L"], "x": x, "obs": obs,
                         "legacy": leg[x], "stls": stl[x], "ls": ls_v, "gauss": g_v})
        print(f"run {run['exp']:>3} {run['pg']}: " + "  ".join(
            f"{x}m obs {run['obs'][x]:.4f} leg {leg[x]:.4f} stls {stl[x]:.4f}" for x in ARCS if x in run["obs"]),
            flush=True)
    with open(os.path.join(HERE, "model_comparison_pg.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print()
    for name in ("legacy", "stls", "ls", "gauss"):
        s = scores([r["obs"] for r in rows], [r[name] for r in rows])
        zeros = sum(1 for r in rows if not r[name] > 0)
        print(f"{name:7s} n={s['n']:3d} zeros={zeros:3d} FB={s['FB']:+.2f} NMSE={s['NMSE']:.2f} "
              f"FAC2={s['FAC2']:.2f} MG={s['MG']:.2f}")


if __name__ == "__main__":
    main()
