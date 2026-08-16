# src/dsai/climatology.py
#
# Station-conditioned exceedance detection: both absolute value AND
# rate-of-change (hour-over-hour delta), each checked against that
# station's own historical percentile distribution rather than a fixed
# threshold. Proven against the real 2026-07-18 Edmonton East H2S event
# (flagged at the 100th percentile on both dimensions).

import json
import os


def fetch_station_series(sb, station, parameter, since_iso, until_iso=None):
    """Pull the full hourly time series for one station/parameter, paginated."""
    rows = []
    page_size = 1000
    offset = 0
    while True:
        q = (
            sb.table("aqhi_data")
            .select("ReadingDate,Value")
            .eq("StationName", station)
            .eq("ParameterName", parameter)
            .gte("ReadingDate", since_iso)
            .order("ReadingDate")
            .range(offset, offset + page_size - 1)
        )
        if until_iso:
            q = q.lte("ReadingDate", until_iso)
        res = q.execute()
        if not res.data:
            break
        rows.extend(res.data)
        if len(res.data) < page_size:
            break
        offset += page_size
    return [
        (r["ReadingDate"], r["Value"])
        for r in rows
        if r["Value"] is not None
    ]


def build_climatology(series):
    """
    series: list of (timestamp_iso, value), ascending by time.
    Returns sorted values and sorted hour-over-hour deltas, ready for
    percentile lookups.
    """
    values = [v for _, v in series]
    deltas = []
    for i in range(1, len(series)):
        deltas.append(series[i][1] - series[i - 1][1])

    return {
        "values_sorted": sorted(values),
        "deltas_sorted": sorted(deltas),
        "n_values": len(values),
        "n_deltas": len(deltas),
        "last_timestamp": series[-1][0] if series else None,
    }


def value_percentile_rank(clim, value):
    sv = clim["values_sorted"]
    if not sv:
        return None
    below = sum(1 for v in sv if v <= value)
    return 100.0 * below / len(sv)


def delta_percentile_rank(clim, delta):
    sd = clim["deltas_sorted"]
    if not sd:
        return None
    below = sum(1 for d in sd if d <= delta)
    return 100.0 * below / len(sd)


def check_exceedance(clim, current_value, current_delta, abs_pct_threshold=95, roc_pct_threshold=95):
    abs_rank = value_percentile_rank(clim, current_value)
    roc_rank = delta_percentile_rank(clim, current_delta)

    abs_flag = abs_rank is not None and abs_rank >= abs_pct_threshold
    roc_flag = roc_rank is not None and roc_rank >= roc_pct_threshold

    return {
        "value": current_value,
        "value_percentile": round(abs_rank, 1) if abs_rank is not None else None,
        "value_flag": abs_flag,
        "delta": current_delta,
        "delta_percentile": round(roc_rank, 1) if roc_rank is not None else None,
        "delta_flag": roc_flag,
        "any_flag": abs_flag or roc_flag,
    }


# ---------------------------
# Cache persistence
# ---------------------------

CACHE_PATH = "/opt/airquality/dsai_data/climatology_cache.json"


def save_cache(cache, path=CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)


def load_cache(path=CACHE_PATH):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def cache_key(station, parameter):
    return f"{station}::{parameter}"
