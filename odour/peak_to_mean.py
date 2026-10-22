"""
Peak-to-mean factors: hourly-mean model concentrations -> what a nose perceives.

Dispersion models (this LS model, the Gaussian puff, AERMOD) give hour-average concentrations,
but odour is perceived over about a second, and short-term peaks are several times the hourly mean.
Two regulatory methods are implemented, so a result can be stated under either:

1. NSW EPA, "Approved Methods for the Modelling and Assessment of Air Pollutants in New South Wales"
   (2022), Section 6.6, Table 10 (from Katestone Scientific 1995, 1998): ratio of the peak 1-second
   average to the mean 1-hour average (P/M60), by source type, Pasquill-Gifford class, and near- or
   far-field receptor. NSW applies these to H2S and complex odour mixtures. The table is for one
   source in flat terrain with the receptor on the plume centreline.

2. Ontario O. Reg. 419/05 s.17 / MECP "Methodology for modelling assessments of contaminants with
   10-minute average standards" (2016): C_10min = C_1h * (60/10)**0.28 = 1.65 x. Ontario's TRS, H2S
   and mercaptan odour standards are 10-minute averages.

NSW does not define near- vs far-field in the Approved Methods. NEAR_FIELD_FACTOR below (receptor
closer than 10 x the source's largest horizontal dimension = near field) is OUR assumption, not NSW's.
"""

ONTARIO_EXPONENT = 0.28

# NSW Approved Methods (2022) Table 10: {source type: {(classes): (near-field, far-field)}}
NSW_TABLE10 = {
    "area":                 {"ABCD": (2.5, 2.3), "EF": (2.3, 1.9)},
    "line":                 {"ABCDEF": (6.0, 6.0)},
    "surface_point":        {"ABC": (12.0, 4.0), "DEF": (25.0, 7.0)},    # wake-free
    "tall_point":           {"ABC": (17.0, 3.0), "DEF": (35.0, 6.0)},    # wake-free
    "wake_affected_point":  {"ABCDEF": (2.3, 2.3)},
    "volume":               {"ABCDEF": (2.3, 2.3)},
}
NEAR_FIELD_FACTOR = 10.0   # assumption, see module docstring


def ontario_factor(t_short_min=10.0, t_long_min=60.0):
    """Ontario power law: C_short = C_long * (t_long / t_short) ** 0.28 (10 min from 1 h = 1.65)."""
    return (t_long_min / t_short_min) ** ONTARIO_EXPONENT


def is_near_field(distance_m, source_size_m):
    return distance_m < NEAR_FIELD_FACTOR * max(source_size_m, 1.0)


def nsw_factor(source_type, pg_class, near_field):
    """NSW P/M60 (peak 1 s / mean 1 h) for a source type, P-G class (A-F) and near/far field."""
    rows = NSW_TABLE10[source_type]
    for classes, (near, far) in rows.items():
        if pg_class.upper() in classes:
            return near if near_field else far
    raise ValueError(f"no NSW P/M60 for {source_type!r} class {pg_class!r}")


def dominant_class(pg_classes):
    """Most frequent P-G class over a run; ties go to the more stable class (higher P/M60 for
    point sources, so the conservative choice)."""
    counts = {}
    for c in pg_classes:
        counts[c] = counts.get(c, 0) + 1
    return max(counts, key=lambda c: (counts[c], c))


def annotate(rows, pg_class, source_type="wake_affected_point", source_size_m=100.0):
    """Add peak estimates to source-ranking rows that carry cq_s_per_m3 (hour-mean C/Q, s/m3) and
    distance_km. Adds cq_peak_1s_nsw, cq_10min_ontario and the factors used."""
    f_on = ontario_factor()
    for r in rows:
        near = is_near_field(r.get("distance_km", 0.0) * 1000.0, source_size_m)
        f_nsw = nsw_factor(source_type, pg_class, near)
        r["pm_factor_nsw_1s"] = f_nsw
        r["pm_factor_ontario_10min"] = round(f_on, 3)
        r["cq_peak_1s_nsw"] = r["cq_s_per_m3"] * f_nsw
        r["cq_10min_ontario"] = r["cq_s_per_m3"] * f_on
    return rows


def describe(pg_class, source_type="wake_affected_point", source_size_m=100.0):
    return {
        "nsw": {"ref": "NSW EPA Approved Methods (2022) Table 10, P/M60 = peak 1 s / mean 1 h",
                "source_type": source_type, "pg_class": pg_class,
                "near_field_factor": NEAR_FIELD_FACTOR, "source_size_m": source_size_m,
                "near_field_rule": "assumption: near field if distance < 10 x source size (NSW gives no definition)"},
        "ontario": {"ref": "O. Reg. 419/05 s.17; MECP 10-minute methodology (2016)",
                    "formula": "C_10min = C_1h * (60/10)^0.28", "factor": round(ontario_factor(), 3)},
    }


# ---------------------------------------------------------------------------
# Estimated concentrations from NPRI releases (2026-10-22, batch-2 item 7)
# ---------------------------------------------------------------------------
NPRI_RELEASES = "/opt/airquality/dsai_data/npri_air_releases.json"   # built by dsai/build_npri_releases.py
MW = {"H2S": 34.08, "TRS": 34.08, "SO2": 64.07, "NOx": 46.01, "NH3": 17.03, "CO": 28.01, "Benzene": 78.11}
# Alberta Ambient Air Quality Objectives, 1-hour, ppb (NOx is compared as if all NO2 - an upper bound)
AAAQO_1H_PPB = {"H2S": 10.0, "SO2": 172.0, "NOx": 159.0}
_REL = None


def _releases():
    global _REL
    if _REL is None:
        try:
            import json
            with open(NPRI_RELEASES) as fh:
                _REL = json.load(fh).get("facilities", {})
        except (OSError, ValueError):
            _REL = {}
    return _REL


def estimate_concentrations(rows):
    """For ranking rows with npri_id and cq_s_per_m3 (hour-mean C/Q), add "est": per substance the annual-
    average release (g/s) times C/Q = estimated hour-mean concentration at the receptor (ug/m3; ppb for
    gases at 25 C, 1 atm), its NSW 1 s peak and Ontario 10 min value, and the AAAQO 1 h objective where one
    exists. Annual-average emissions: real hourly releases vary, so this is a screening estimate."""
    rel = _releases()
    for r in rows:
        f = rel.get(str(r.get("npri_id")))
        if not f:
            continue
        est = {}
        for sub, x in f["substances"].items():
            c = r["cq_s_per_m3"] * x["g_per_s"] * 1e6           # ug/m3, hour mean
            e = {"g_per_s": x["g_per_s"], "ug_m3": round(c, 4),
                 "ug_m3_peak_1s": round(c * r.get("pm_factor_nsw_1s", 1.0), 4),
                 "ug_m3_10min": round(c * r.get("pm_factor_ontario_10min", 1.0), 4)}
            if sub in MW:
                ppb = c * 24.45 / MW[sub]
                e["ppb"] = round(ppb, 4)
                e["ppb_peak_1s"] = round(ppb * r.get("pm_factor_nsw_1s", 1.0), 4)
                if sub in AAAQO_1H_PPB:
                    e["aaaqo_1h_ppb"] = AAAQO_1H_PPB[sub]
            est[sub] = e
        r["est"] = est
        r["npri_year"] = f["year"]
    return rows
