# src/dsai/compare_trajectories.py
#
# Real answer to "now we have a dual model check - any way to compare
# the outputs?" (2026-09-13). DSAI's HYSPLIT ensemble (GFS met, 111km
# grid, no terrain) and AB_winds' HRDPS particle model
# (odour/backtraj_core.py - HRDPS met, ~2.5km grid, DEM terrain
# steering) both produce a backward trajectory for the same
# station/event - this loads both and reports how far apart they put
# the air parcel at each matched hour back, plus a combined GeoJSON for
# visual overlay.
#
# Real methodology note, not swept under the rug: the two models don't
# share release heights. HYSPLIT runs 100/500/1000m AGL; HRDPS
# (odour_index's DEFAULT_HEIGHTS) runs 10/40/80m AGL. The nearest pair
# - HYSPLIT 100m vs HRDPS 80m - is used as the primary comparison since
# both are the lowest, most near-surface-transport-relevant release
# height each model offers, not because they're the same height.
# Genuine agreement there is meaningful; a mismatch could be either a
# real model disagreement or partly an artifact of comparing 100m to
# 80m - keep that in mind before reading too much into a gap.

import json
import math
import os

PRIMARY_HYSPLIT_HEIGHT = 100
PRIMARY_HRDPS_HEIGHT = 80.0
HRDPS_DT_S = 60  # odour_index.py's DEFAULT_DT - seconds between HRDPS particle steps

EARTH_R_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_R_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_tdump(tdump_path):
    """Returns {age_hours (negative int): (lat, lon, height_m)} for one
    HYSPLIT tdump file. Header length is variable (depends how many met
    files were used) - real trajectory rows are recognized by column
    count (13) and a parseable age-hours field, not a fixed line number."""
    points = {}
    with open(tdump_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 13:
                continue
            try:
                age_hours = float(parts[8])
                lat = float(parts[9])
                lon = float(parts[10])
                height_m = float(parts[11])
            except ValueError:
                continue
            points[round(age_hours)] = (lat, lon, height_m)
    return points


def parse_hrdps_centerline(geojson_path, height_m=PRIMARY_HRDPS_HEIGHT):
    """Returns {age_hours (negative int): (lat, lon)} for one height's
    centerline from an AB_winds odour back-trajectory run. Coordinate
    index 0 is the event start; index N is N*HRDPS_DT_S seconds back -
    resampled here to whole hours to match tdump's native resolution."""
    with open(geojson_path) as fh:
        data = json.load(fh)
    for feat in data["features"]:
        if feat["properties"].get("z0_m") == height_m:
            coords = feat["geometry"]["coordinates"]
            points = {}
            for i, (lon, lat) in enumerate(coords):
                elapsed_s = i * HRDPS_DT_S
                age_hours = round(elapsed_s / 3600.0)
                # first point wins at each whole hour - coords are already
                # ordered oldest-index-last, so earlier i = closer to "now"
                if age_hours not in points:
                    points[-age_hours] = (lat, lon)
            return points
    raise ValueError(f"No {height_m}m centerline in {geojson_path}")


def compare(hysplit_work_dir, hrdps_centerline_path,
            hysplit_height=PRIMARY_HYSPLIT_HEIGHT, hrdps_height=PRIMARY_HRDPS_HEIGHT):
    """
    Returns a list of {age_hours, hysplit_latlon, hrdps_latlon,
    divergence_km} for every whole hour both models cover, plus the two
    full point-sets (all matched heights) for building a combined
    GeoJSON overlay.
    """
    tdump_path = os.path.join(hysplit_work_dir, f"tdump_{hysplit_height}m")
    hysplit_pts = parse_tdump(tdump_path)
    hrdps_pts = parse_hrdps_centerline(hrdps_centerline_path, height_m=hrdps_height)

    rows = []
    for age in sorted(set(hysplit_pts) & set(hrdps_pts), reverse=True):
        hlat, hlon, _ = hysplit_pts[age]
        rlat, rlon = hrdps_pts[age]
        rows.append({
            "age_hours": age,
            "hysplit_latlon": (hlat, hlon),
            "hrdps_latlon": (rlat, rlon),
            "divergence_km": round(haversine_km(hlat, hlon, rlat, rlon), 1),
        })
    return rows


def combined_geojson(hysplit_work_dir, hrdps_centerline_path,
                      hysplit_heights=(100, 500, 1000), hrdps_heights=(10.0, 40.0, 80.0)):
    """Both models' full trajectories (all heights), as one
    FeatureCollection for a single-map overlay - model + height on each
    feature so a legend/style function can tell them apart."""
    features = []
    for h in hysplit_heights:
        tdump_path = os.path.join(hysplit_work_dir, f"tdump_{h}m")
        if not os.path.exists(tdump_path):
            continue
        pts = parse_tdump(tdump_path)
        if not pts:
            continue
        coords = [[lon, lat] for _, (lat, lon, _) in sorted(pts.items(), reverse=True)]
        features.append({
            "type": "Feature",
            "properties": {"model": "HYSPLIT (GFS)", "height_m": h},
            "geometry": {"type": "LineString", "coordinates": coords},
        })

    with open(hrdps_centerline_path) as fh:
        hrdps_data = json.load(fh)
    for feat in hrdps_data["features"]:
        z0 = feat["properties"].get("z0_m")
        if z0 in hrdps_heights:
            features.append({
                "type": "Feature",
                "properties": {"model": "HRDPS (particle, terrain-steered)", "height_m": z0},
                "geometry": feat["geometry"],
            })

    return {"type": "FeatureCollection", "features": features}


def print_report(rows, station):
    if not rows:
        print(f"No overlapping hours between the two models for {station} - nothing to compare.")
        return
    print(f"Dual-model trajectory comparison for {station} "
          f"(HYSPLIT {PRIMARY_HYSPLIT_HEIGHT}m AGL vs HRDPS {PRIMARY_HRDPS_HEIGHT:.0f}m AGL):")
    print(f"{'hours back':>10} | {'HYSPLIT lat,lon':>22} | {'HRDPS lat,lon':>22} | {'divergence':>10}")
    for r in rows:
        h_lat, h_lon = r["hysplit_latlon"]
        r_lat, r_lon = r["hrdps_latlon"]
        print(f"{-r['age_hours']:>10} | {h_lat:>9.4f},{h_lon:>10.4f} | {r_lat:>9.4f},{r_lon:>10.4f} | {r['divergence_km']:>8.1f}km")
    max_div = max(r["divergence_km"] for r in rows)
    print(f"\nMax divergence over the overlap window: {max_div:.1f}km")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: compare_trajectories.py <station> <hysplit_work_dir> <hrdps_centerline_geojson>")
        sys.exit(1)

    station, hysplit_dir, hrdps_path = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = compare(hysplit_dir, hrdps_path)
    print_report(rows, station)

    geo = combined_geojson(hysplit_dir, hrdps_path)
    out_path = os.path.join(hysplit_dir, "dual_model_comparison.geojson")
    with open(out_path, "w") as fh:
        json.dump(geo, fh)
    print(f"\nCombined overlay: {out_path}")
