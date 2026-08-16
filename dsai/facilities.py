# src/dsai/facilities.py
#
# Layer 2: Local Source Likelihood. Given a station and a current (or
# historical) wind direction, finds which real NPRI-registered
# facilities lie in the upwind sector - i.e. bearing FROM the station
# TO the facility falls within the wind's "coming from" direction,
# meaning emissions there would blow toward the station.
#
# Data source: NPRI's live ArcGIS REST endpoint (verified 2026-08-16;
# the old SC_AQMap/NextGen_dk NPRI.geojson reference is a dead link -
# that repo no longer exists). Alberta: 3,266 facilities, refreshed
# periodically via refresh_npri_data() rather than bundled statically,
# since NPRI updates its data annually and facilities open/close.

import json
import math
import os
import urllib.request
import urllib.parse

NPRI_ENDPOINT = "https://maps-cartes.ec.gc.ca/arcgis/rest/services/STB_DGST/NPRI/MapServer/0/query"
NPRI_CACHE_PATH = "/opt/airquality/dsai_data/npri_alberta.geojson"
PAGE_SIZE = 2000


def refresh_npri_data(province="AB", cache_path=NPRI_CACHE_PATH):
    """Pull the full current facility set for a province from NPRI's live endpoint."""
    all_features = []
    offset = 0
    while True:
        params = {
            "where": f"ProvinceCode='{province}'",
            "outFields": "*",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
        }
        url = f"{NPRI_ENDPOINT}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
        feats = data.get("features", [])
        all_features.extend(feats)
        if len(feats) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    merged = {"type": "FeatureCollection", "features": all_features}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(merged, f)
    return len(all_features)


def load_facilities(cache_path=NPRI_CACHE_PATH):
    if not os.path.exists(cache_path):
        refresh_npri_data(cache_path=cache_path)
    with open(cache_path) as f:
        data = json.load(f)
    return data["features"]


def bearing_and_distance(lat1, lon1, lat2, lon2):
    """Bearing (compass degrees, 0-360) and great-circle distance (km) from point 1 to point 2."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    distance_km = 2 * R * math.asin(math.sqrt(a))

    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing_deg = (math.degrees(math.atan2(y, x)) + 360) % 360

    return bearing_deg, distance_km


def angular_diff(a, b):
    """Smallest angular difference between two compass bearings, 0-180."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def facilities_upwind(station_lat, station_lon, wind_from_deg, facilities,
                       max_distance_km=100, sector_width_deg=45):
    """
    Which facilities lie in the sector the wind is currently coming FROM
    (bearing from station to facility ~= wind_from_deg), within
    max_distance_km. Returns list of dicts sorted by distance, each
    with facility info + computed bearing/distance/off-axis degrees.
    """
    results = []
    for feat in facilities:
        props = feat.get("properties", {})
        lat = props.get("Latitude")
        lon = props.get("Longitude")
        if lat is None or lon is None:
            continue

        bearing, distance_km = bearing_and_distance(station_lat, station_lon, lat, lon)
        if distance_km > max_distance_km:
            continue

        off_axis = angular_diff(bearing, wind_from_deg)
        if off_axis > sector_width_deg / 2:
            continue

        results.append({
            "facility_name": props.get("FacilityName"),
            "company_name": props.get("CompanyName"),
            "sector": props.get("SectorDescriptionEn"),
            "naics_code": props.get("NAICS__Code_SCIAN"),
            "report_year": props.get("ReportYear"),
            "distance_km": round(distance_km, 1),
            "bearing_from_station": round(bearing, 1),
            "off_axis_deg": round(off_axis, 1),
        })

    results.sort(key=lambda r: r["distance_km"])
    return results


if __name__ == "__main__":
    # Validate against the real July 18 Edmonton East case: WD=284 at
    # the spike hour. Kevin described the station as sitting "between
    # two refineries" - see if real NPRI data backs that up.
    from stations import STATIONS

    facilities = load_facilities()
    print(f"Loaded {len(facilities)} Alberta NPRI facilities")

    lat, lon = STATIONS["Edmonton East"]
    print(f"\nFacilities upwind of Edmonton East at WD=284 deg, within 100km:")
    upwind = facilities_upwind(lat, lon, wind_from_deg=284, facilities=facilities,
                                max_distance_km=100, sector_width_deg=45)
    for f in upwind[:15]:
        print(f"  {f['distance_km']:6.1f} km, bearing {f['bearing_from_station']:6.1f} "
              f"(off-axis {f['off_axis_deg']:5.1f}) - {f['facility_name']} "
              f"({f['company_name']}) - {f['sector']}")
    print(f"\n{len(upwind)} facilities total in that sector/range")
