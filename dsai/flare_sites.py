# flare_sites.py - active AER facilities (all of Alberta) for the LiveMap "likely gas flare" check (2026-11-20).
#
# Same rule as the PAZA/WCAS emergency maps (paza_emergency_intelligence/flares.py): a small, mostly
# night-time heat detection within 2 km of an active oil & gas facility is labelled a likely flare.
# ST102 is Alberta only, so outside AB the facility part of the rule simply doesn't fire (the same-spot
# rules still do). fetch_firms_map.py calls refresh() when the file is over 30 days old.
#
#   python flare_sites.py      rebuild data/flare_sites.json now

import datetime as dt
import json
import os
import shutil
import tempfile
import time
import zipfile

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
SITES_PATH = os.path.join(HERE, "data", "flare_sites.json")
ST102_URL = "https://www.aer.ca/data/codes/ST102-SHP.zip"
ST102_ZIP = "/opt/airquality/data/aer/ST102-SHP.zip"   # shared with the emergency maps
MAX_AGE = 30 * 86400


def refresh(force=False):
    if not force and os.path.exists(SITES_PATH) and time.time() - os.path.getmtime(SITES_PATH) < MAX_AGE:
        return False
    import geopandas as gpd
    os.makedirs(os.path.dirname(ST102_ZIP), exist_ok=True)
    if not os.path.exists(ST102_ZIP) or time.time() - os.path.getmtime(ST102_ZIP) > MAX_AGE:
        r = requests.get(ST102_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=900)
        r.raise_for_status()
        tmp = ST102_ZIP + ".tmp"
        open(tmp, "wb").write(r.content)
        os.replace(tmp, ST102_ZIP)
    d = tempfile.mkdtemp(prefix="st102_")
    try:
        with zipfile.ZipFile(ST102_ZIP) as zf:
            shp = [n for n in zf.namelist() if n.lower().endswith("gcs_nad83.shp")][0]
            zf.extractall(d)
        g = gpd.read_file(os.path.join(d, shp))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if g.crs and g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    g = g[g.FAC_STATUS.isin(["Active", "New"]) & ~g.FAC_SUB_TY.fillna("").str.contains("Water Source")]
    pt = g.geometry.representative_point()
    out = [[round(p.y, 5), round(p.x, 5), n, s] for p, n, s in zip(pt, g.FAC_NAME, g.FAC_SUB_TY)]
    os.makedirs(os.path.dirname(SITES_PATH), exist_ok=True)
    tmp = SITES_PATH + ".tmp"
    json.dump({"source": f"AER ST102 ({dt.date.fromtimestamp(os.path.getmtime(ST102_ZIP))}), Active/New",
               "sites": out}, open(tmp, "w"))
    os.replace(tmp, SITES_PATH)
    print(f"flare sites: {len(out)} active AER facilities")
    return True


if __name__ == "__main__":
    refresh(force=True)
