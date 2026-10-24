"""Move finished HPFX backfill files (met_data_hpfx/, see hpfx_backfill.py) into met_data/.

New names are moved in as they are. Where met_data already has the file (the
3-hourly leads the live pull kept, on the old grid without the BC coast), the
staged copy replaces it only if every overlapping cell matches (same HRDPS
source and regrid; the backfill just covers more area). Mismatches are left in
met_data_hpfx/conflicts/ and reported. Decoded .npz caches of replaced files are
removed so LocalMetStore rebuilds them with the wider grid. The nightly Dropbox
backup keeps the replaced originals under versions/<date>/.

Safe to run while the backfill is still going: files are written via .part.
Usage: hpfx_merge.py [--dry-run]
"""

import glob
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE, MET = os.path.join(ROOT, "met_data_hpfx"), os.path.join(ROOT, "met_data")
CACHES = ["/opt/airquality/dsai_data/met_npz_cache"]
TOL = 0.011   # values are stored rounded to 0.01


def same_on_overlap(new, old):
    a, b = json.load(gzip.open(new)), json.load(gzip.open(old))
    ga, gb = a["grid"], b["grid"]
    if (ga["dx"], ga["dy"]) != (gb["dx"], gb["dy"]):
        return False, "different grid spacing"
    dj = round((gb["lo1"] - ga["lo1"]) / ga["dx"]); di = round((ga["la1"] - gb["la1"]) / ga["dy"])
    for k in ("ugrd10", "vgrd10", "tmp2", "tmp40"):
        if k not in a["fields"] or k not in b["fields"]:
            continue
        x = np.array(a["fields"][k][0], dtype=float); y = np.array(b["fields"][k][0], dtype=float)
        sub = x[di:di + y.shape[0], dj:dj + y.shape[1]]
        if sub.shape != y.shape:
            return False, f"{k}: old grid not inside new"
        m = np.isfinite(sub) & np.isfinite(y)
        if m.sum() == 0 or np.max(np.abs(sub[m] - y[m])) > TOL:
            return False, f"{k}: values differ"
    return True, ""


def main(dry):
    moved = replaced = conflicts = 0
    os.makedirs(os.path.join(STAGE, "conflicts"), exist_ok=True)
    for p in sorted(glob.glob(os.path.join(STAGE, "ab_met_*.json.gz"))):
        name = os.path.basename(p); dst = os.path.join(MET, name)
        if not os.path.exists(dst):
            if not dry: os.replace(p, dst)
            moved += 1; continue
        ok, why = same_on_overlap(p, dst)
        if not ok:
            print(f"conflict {name}: {why}")
            if not dry: os.replace(p, os.path.join(STAGE, "conflicts", name))
            conflicts += 1; continue
        if not dry:
            os.replace(p, dst)
            for c in CACHES:
                for f in glob.glob(os.path.join(c, name.replace(".json.gz", "") + "*")):
                    os.remove(f)
        replaced += 1
    print(f"{'(dry run) ' if dry else ''}moved {moved} new, replaced {replaced} (wider grid, identical overlap), conflicts {conflicts}")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
