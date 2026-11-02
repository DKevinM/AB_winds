"""Crosswind-integrated concentration (CWIC) profiles: Project Sagebrush measured vs forward LS model.

Reads validation/sagebrush/sagebrush_results.csv (from validate_sagebrush.py) and, optionally, an older
results file to overlay (e.g. the pre-2026-10-22 run with SIGMA_FLOOR = 0.05):
    python validation/plot_sagebrush_cwic.py out.png [old_results.csv]

Normalised as Cy*U/Q (1/m) so periods with different wind speeds sit on one scale.
Usable arc-periods = same filter as validate_sagebrush.py (not truncated, max excess >= 20 ppt, obs > 0).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = __file__.rsplit("/", 1)[0]
OUT = sys.argv[1]
OLD = sys.argv[2] if len(sys.argv) > 2 else None

GROUPS = [("PSB1 2013, day", "#2a78d6"), ("PSB2 2016, day", "#eb6834"), ("PSB2 2016, night (very stable)", "#1baf7a")]


def load(path):
    d = pd.read_csv(path)
    d = d[(~d.truncated) & (d.max_excess_ppt >= 20) & (d.cyq_obs > 0) & (d.cyq_ls > 0)].copy()
    d["grp"] = np.where(d.phase == "PSB1", GROUPS[0][0], np.where(d.iop <= 4, GROUPS[1][0], GROUPS[2][0]))
    d["obs"] = d.cyq_obs * d.U
    d["mod"] = d.cyq_ls * d.U
    return d


def stats(o, m):
    o, m = np.asarray(o), np.asarray(m)
    r = m / o
    fb = 2 * (o.mean() - m.mean()) / (o.mean() + m.mean())
    return dict(n=len(o), fac2=np.mean((r >= 0.5) & (r <= 2)), fb=fb, mg=np.exp(np.mean(np.log(o / m))))


d = load(f"{HERE}/sagebrush/sagebrush_results.csv")
old = load(OLD) if OLD else None

fig = plt.figure(figsize=(14, 9.6), dpi=140)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.38, wspace=0.26)
ink, muted, grid = "#1d2328", "#5b6670", "#e3e7ea"

for k, (g, col) in enumerate(GROUPS):
    ax = fig.add_subplot(gs[0, k])
    s = d[d.grp == g]
    ax.scatter(s.dist * np.exp(np.random.default_rng(1).normal(0, 0.04, len(s))), s.obs, s=14, color=col, alpha=0.28,
               edgecolors="none", label="measured, each 10-min arc")
    agg = s.groupby("dist").agg(om=("obs", "median"), o25=("obs", lambda x: x.quantile(.25)),
                                o75=("obs", lambda x: x.quantile(.75)), mm=("mod", "median"), n=("obs", "size"))
    ax.errorbar(agg.index, agg.om, yerr=[agg.om - agg.o25, agg.o75 - agg.om], fmt="o", ms=8, color=col,
                mec="white", mew=1.5, ecolor=col, elinewidth=1.6, capsize=0, zorder=5, label="measured median (IQR)")
    ax.plot(agg.index, agg.mm, "-", color=ink, lw=2, marker="s", ms=5, zorder=6, label="LS model median (current)")
    if old is not None and "night" in g:
        so = old[old.grp == g].groupby("dist")["mod"].median()
        ax.plot(so.index, so.values, "--", color=muted, lw=1.6, zorder=4, label="LS model before night fix")
    st = stats(s.cyq_obs, s.cyq_ls)
    ax.set_title(g, fontsize=12, loc="left", color=ink, fontweight="bold")
    ax.text(0.03, 0.04, f"n = {st['n']}   within factor of 2: {st['fac2']:.0%}\nfractional bias {st['fb']:+.2f}",
            transform=ax.transAxes, fontsize=9.5, color=ink, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ticks = [t for t in (100, 200, 400, 800, 1600, 3200) if t in set(s.dist)]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:,}" for t in ticks]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(min(ticks) / 1.35, max(ticks) * 1.35)
    ax.set_xlabel("Distance downwind (m)", fontsize=10, color=muted)
    if k == 0:
        ax.set_ylabel("Crosswind-integrated  Cy·U/Q  (1/m)", fontsize=10, color=muted)
    ax.grid(True, which="major", color=grid, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#b9c1c7")
    ax.tick_params(colors=muted, labelsize=9)
    if k == 0:
        h, l = ax.get_legend_handles_labels()

# Scatter: every usable arc-period, model vs measured
ax = fig.add_subplot(gs[1, 0:2])
lo, hi = 10 ** np.floor(np.log10(min(d.obs.min(), d["mod"].min()))), 10 ** np.ceil(np.log10(max(d.obs.max(), d["mod"].max())))
xx = np.array([lo, hi])
ax.fill_between(xx, xx / 2, xx * 2, color="#eef2f5", zorder=0, label="within factor of 2")
ax.plot(xx, xx, color="#9aa6af", lw=1, zorder=1)
for g, col in GROUPS:
    s = d[d.grp == g]
    ax.scatter(s.obs, s["mod"], s=18, color=col, alpha=0.7, edgecolors="white", linewidths=0.4, label=g, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Measured  Cy·U/Q  (1/m)", fontsize=10, color=muted)
ax.set_ylabel("LS model  Cy·U/Q  (1/m)", fontsize=10, color=muted)
ax.set_title("Every arc-period: model vs measured", fontsize=12, loc="left", color=ink, fontweight="bold")
ax.grid(True, color=grid, lw=0.8); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.tick_params(colors=muted, labelsize=9)
ax.legend(fontsize=9, frameon=False, loc="upper left")

# Summary table
ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
rows = [("Group", "n", "FAC2", "FB", "MG")]
for g, _ in GROUPS:
    s = d[d.grp == g]; st = stats(s.cyq_obs, s.cyq_ls)
    rows.append((g.replace(" (very stable)", ""), st["n"], f"{st['fac2']:.0%}", f"{st['fb']:+.2f}", f"{st['mg']:.2f}"))
st = stats(d.cyq_obs, d.cyq_ls); rows.append(("All", st["n"], f"{st['fac2']:.0%}", f"{st['fb']:+.2f}", f"{st['mg']:.2f}"))
if old is not None:
    so = old[old.grp == GROUPS[2][0]]; st = stats(so.cyq_obs, so.cyq_ls)
    rows.append(("Night, before fix", st["n"], f"{st['fac2']:.0%}", f"{st['fb']:+.2f}", f"{st['mg']:.2f}"))
tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper center", cellLoc="center", colWidths=[0.46, 0.12, 0.14, 0.14, 0.14])
tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.6)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#dde2e6")
    if r == 0: cell.set_text_props(fontweight="bold", color=ink); cell.set_facecolor("#f3f5f7")
ax.text(0, 0.30, "FAC2: share within a factor of 2 (good: ≥ 50%)\nFB: fractional bias, + = model low (good: |FB| < 0.3)\n"
        "MG: geometric mean bias, obs/model (good: 0.7–1.3)\nAcceptance ranges: Chang & Hanna (2004)",
        transform=ax.transAxes, fontsize=8.8, color=muted, va="top", linespacing=1.5)

fig.suptitle("Project Sagebrush: crosswind-integrated concentrations, measured vs LS model", x=0.06, ha="left",
             fontsize=16, fontweight="bold", color=ink, y=0.975)
fig.text(0.06, 0.925, "SF6 tracer released at 1.5 m, 10-minute bag samples at 1 m on arcs 100–3200 m (NOAA ARL, Idaho National Laboratory). "
         "Model driven by each period's measured u*, L and wind speed.", fontsize=10, color=muted)
from matplotlib.lines import Line2D
leg = [Line2D([], [], marker="o", ls="", color="#7d8a94", alpha=0.35, ms=5, label="measured, each 10-min arc"),
       Line2D([], [], marker="o", ls="", color="#7d8a94", mec="white", ms=8, label="measured median, bar = middle 50%"),
       Line2D([], [], marker="s", color=ink, lw=2, ms=5, label="LS model median (current)")]
if old is not None:
    leg.append(Line2D([], [], ls="--", color=muted, lw=1.6, label="LS model before night fix (night panel)"))
fig.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.06, 0.905), ncol=len(leg), frameon=False, fontsize=9.5)
fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.07)
fig.savefig(OUT, facecolor="white")
print("saved", OUT)
