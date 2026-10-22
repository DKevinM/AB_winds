# src/dsai/build_npri_releases.py
#
# Facility releases to AIR from the NPRI bulk file, for turning the LS/puff source ranking's C/Q (s/m3)
# into an estimated concentration at the receptor (2026-10-22, batch-2 item 7).
#
# Source: ECCC NPRI bulk data, "NPRI-INRP_ReleasesRejets_1993-present.csv" (open.canada.ca dataset
# 40e01423-7728-429c-ac9d-2954385ccdfb). The ECCC data mart page is now a JavaScript app; the file
# itself downloads from the Azure file API:
#   https://data-donnees.az.ec.gc.ca/api/file?path=/substances/plansreports/reporting-facilities-pollutant-
#   release-and-transfer-data/bulk-data-files-for-all-years-releases-disposals-transfers-and-facility-
#   locations/NPRI-INRP_ReleasesRejets_1993-present.csv
# NPRI is annual; re-run once a year after the new reporting year is published (usually spring).
#
# Output /opt/airquality/dsai_data/npri_air_releases.json:
#   {"source": ..., "built": ..., "facilities": {NPRI_ID: {"year", "name", "company", "province",
#     "substances": {key: {"t_per_yr", "g_per_s", "stack_share", "road_dust_t"}}}}}
# g_per_s is the ANNUAL AVERAGE rate (t/yr / seconds per year) - real hourly emissions vary.
#
# Usage: python build_npri_releases.py [path to the bulk csv]   (downloads it if no path given)

import datetime as dt
import json
import os
import subprocess
import sys

import pandas as pd

BULK_URL = ("https://data-donnees.az.ec.gc.ca/api/file?path=/substances/plansreports/reporting-facilities-"
            "pollutant-release-and-transfer-data/bulk-data-files-for-all-years-releases-disposals-transfers-"
            "and-facility-locations/NPRI-INRP_ReleasesRejets_1993-present.csv")
DEFAULT_CSV = "/opt/airquality/data/npri/releases_1993-present.csv"
OUT = "/opt/airquality/dsai_data/npri_air_releases.json"
PROVINCES = ("AB", "BC", "SK")
MIN_YEAR = 2020           # facilities that stopped reporting before this are left out
SUBSTANCES = {            # our key -> NPRI English substance name
    "H2S": "Hydrogen sulphide",
    "TRS": "Total reduced sulphur (expressed as hydrogen sulphide)",
    "SO2": "Sulphur dioxide",
    "NOx": "Nitrogen oxides (expressed as nitrogen dioxide)",
    "PM2.5": "PM2.5 - Particulate Matter <= 2.5 Micrometers",
    "PM10": "PM10 - Particulate Matter <= 10 Micrometers",
    "TPM": "Total particulate matter",
    "VOC": "Volatile Organic Compounds (Total)",
    "CO": "Carbon monoxide",
    "NH3": "Ammonia (total)",
    "Benzene": "Benzene",
}
TO_TONNES = {"tonnes": 1.0, "kg": 1e-3, "grams": 1e-6}
SEC_PER_YR = 365.25 * 24 * 3600


def main(path=None):
    if path is None:
        path = DEFAULT_CSV
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(["curl", "-sL", "--max-time", "1800", "-o", path, BULK_URL], check=True)
    d = pd.read_csv(path, encoding="latin1", low_memory=False)
    d.columns = [c.split(" / ")[0] for c in d.columns]
    d = d[d.PROVINCE.isin(PROVINCES) & (d["Group (English)"] == "Releases to Air")
          & d["Substance Name (English)"].isin(SUBSTANCES.values()) & (d.Reporting_Year >= MIN_YEAR)]
    d = d[d.Units.isin(TO_TONNES)]
    d["t"] = d.Quantity.astype(float) * d.Units.map(TO_TONNES)
    # each facility's latest reporting year
    last = d.groupby("NPRI_ID").Reporting_Year.transform("max")
    d = d[d.Reporting_Year == last]
    key_of = {v: k for k, v in SUBSTANCES.items()}
    fac = {}
    for (nid, sub), g in d.groupby(["NPRI_ID", "Substance Name (English)"]):
        r0 = g.iloc[0]
        f = fac.setdefault(str(int(nid)), {"year": int(r0.Reporting_Year), "name": r0.Facility_Name,
                                            "company": r0.Company_Name, "province": r0.PROVINCE, "substances": {}})
        road = float(g.loc[g["Category (English)"] == "Road dust", "t"].sum())
        tot = float(g.t.sum()) - road          # road dust is kept separate (diffuse, not from the plant)
        stack = float(g.loc[g["Category (English)"] == "Stack / Point", "t"].sum())
        if tot <= 0 and road <= 0:
            continue
        f["substances"][key_of[sub]] = {"t_per_yr": round(tot, 4), "g_per_s": round(tot * 1e6 / SEC_PER_YR, 6),
                                        "stack_share": round(stack / tot, 3) if tot > 0 else None,
                                        "road_dust_t": round(road, 3) if road else 0.0}
    fac = {k: v for k, v in fac.items() if v["substances"]}
    out = {"source": "ECCC NPRI bulk releases 1993-present (Releases to Air), latest reporting year per facility",
           "url": BULK_URL, "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "provinces": PROVINCES, "min_year": MIN_YEAR, "units": "t_per_yr; g_per_s = annual average",
           "facilities": fac}
    with open(OUT, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    yrs = pd.Series([v["year"] for v in fac.values()]).value_counts().sort_index()
    print(f"wrote {OUT}: {len(fac)} facilities; reporting years {yrs.to_dict()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
