# src/dsai/receptors.py
#
# Layer 2b: Downwind Vulnerable Receptors. Inverse of facilities.py's
# upwind industrial-source logic - instead of "what's upwind that could
# be causing this," asks "what's downwind that this could be reaching."
#
# Schools, hospitals, and senior/long-term care facilities (AB+SK),
# from OpenStreetMap via the free Overpass API - see
# build_receptors_cache.py for why OSM won out over government sources
# (Alberta's own hospital/long-term-care data is PDF-only with no
# coordinates, and stale). `type` on each record is "School",
# "Hospital", or "Senior Care".

import json
import os

from facilities import bearing_and_distance, angular_diff

RECEPTORS_CACHE_PATH = "/opt/airquality/dsai_data/receptors_schools_ab_sk.json"


def load_receptors(cache_path=RECEPTORS_CACHE_PATH):
    if not os.path.exists(cache_path):
        return []
    with open(cache_path) as f:
        return json.load(f)


def receptors_downwind(station_lat, station_lon, wind_from_deg, receptors,
                        max_distance_km=50, sector_width_deg=45):
    """
    Which receptors (schools) lie in the direction the wind is blowing
    TOWARD (bearing from station to receptor ~= wind_from_deg + 180),
    within max_distance_km. Mirrors facilities_upwind()'s sector-match
    logic exactly, just pointed the opposite way - a plume moves AWAY
    from the direction the wind comes from, not toward it.

    max_distance_km defaults tighter than facilities_upwind's 100km -
    a school 100km away isn't a meaningful near-surface exposure
    concern the way an upwind source that far away can still be for
    long-range transport (see project_dsai_hysplit's July 18 case).
    """
    downwind_bearing = (wind_from_deg + 180) % 360

    results = []
    for r in receptors:
        lat, lon = r.get("lat"), r.get("lon")
        if lat is None or lon is None:
            continue

        bearing, distance_km = bearing_and_distance(station_lat, station_lon, lat, lon)
        if distance_km > max_distance_km:
            continue

        off_axis = angular_diff(bearing, downwind_bearing)
        if off_axis > sector_width_deg / 2:
            continue

        results.append({
            "name": r.get("name"),
            "type": r.get("type"),
            "province": r.get("province"),
            "distance_km": round(distance_km, 1),
            "bearing_from_station": round(bearing, 1),
            "off_axis_deg": round(off_axis, 1),
        })

    results.sort(key=lambda x: x["distance_km"])
    return results
