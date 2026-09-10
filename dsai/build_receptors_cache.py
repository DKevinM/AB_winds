# src/dsai/build_receptors_cache.py
#
# One-time/occasional builder for receptors.py's downwind-receptor
# cache - NOT a cron job. Source is OpenStreetMap via the free Overpass
# API (no key, no signup, ODbL-licensed) - checked against StatCan's
# ODEF dataset first (a government schools-only source) but OSM turned
# out to be the better call on every axis: it covers all three receptor
# types in one query (schools, hospitals, AND senior/long-term care -
# `social_facility:for=senior` cleanly isolates nursing homes/assisted
# living/retirement homes from OSM's much broader "social_facility"
# bucket, which also includes food banks, shelters, daycares, etc. -
# those are deliberately excluded here), it's live/current rather than
# a static 2019-2022 snapshot, and Alberta's own government data for
# hospitals (PDF, dated 2018) and long-term care (PDF, no coordinates)
# had no usable equivalent at all.
#
# Rerun by hand occasionally to pick up new/closed facilities - OSM
# coverage only improves over time as more of it gets mapped, there's
# no schedule this needs to follow.

import json
import math
import os
import time

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
PROVINCES = ["CA-AB", "CA-SK"]
OUT_PATH = "/opt/airquality/dsai_data/receptors_schools_ab_sk.json"


def build_query(iso):
    return f"""
    [out:json][timeout:120];
    area["ISO3166-2"="{iso}"]->.a;
    (
      nwr["amenity"="school"](area.a);
      nwr["amenity"="hospital"](area.a);
      nwr["amenity"="social_facility"]["social_facility:for"="senior"](area.a);
    );
    out center tags;
    """


def receptor_type(tags):
    amenity = tags.get("amenity")
    if amenity == "school":
        return "School"
    if amenity == "hospital":
        return "Hospital"
    if amenity == "social_facility":
        return "Senior Care"
    return amenity


HEADERS = {"User-Agent": "KRM-Environmental-DSAI-Receptor-Builder/1.0 (kevin@krmenvironmental.com)"}


MERGE_DISTANCE_M = 75
# Real OSM pattern (confirmed directly on a Sturgeon Lake nursing home,
# 2026-09-10): the same physical facility sometimes gets mapped as two
# separate ways - one for the site/grounds, one for just the building
# footprint (tagged `building=yes` in addition to the same amenity tag)
# - a few metres to a few dozen metres apart, both matching this
# script's query, so it shows up twice in any "downwind receptors"
# list. Not a query bug (both ways really do carry the matching tag in
# OSM) and not something to fix by tightening the query - genuinely
# distinct nearby buildings (e.g. a second school ~500m from another)
# need to stay separate. So this merges same-type receptors within
# MERGE_DISTANCE_M of each other ONLY when it's safe to assume they're
# the same site: identical name, or at least one is unnamed (an
# unnamed way sitting right on top of a named one is almost always the
# building-footprint duplicate of that named site). Two DIFFERENTLY
# named receptors that happen to be close together are left alone -
# that's a real judgment call, not something to silently collapse.
def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _should_merge(a, b):
    if a["type"] != b["type"]:
        return False
    if _haversine_m(a["lat"], a["lon"], b["lat"], b["lon"]) > MERGE_DISTANCE_M:
        return False
    return a["name"] == b["name"] or a["name"] == "(unnamed)" or b["name"] == "(unnamed)"


def dedupe_nearby(receptors):
    kept = []
    merged_count = 0
    for r in receptors:
        match = next((k for k in kept if _should_merge(k, r)), None)
        if match is None:
            kept.append(r)
            continue
        merged_count += 1
        if match["name"] == "(unnamed)" and r["name"] != "(unnamed)":
            match.update(r)  # prefer the named one's fields going forward
    if merged_count:
        print(f"Merged {merged_count} near-duplicate receptor(s) (same type, within {MERGE_DISTANCE_M}m, unnamed-or-matching-name)")
    return kept


def fetch_province(iso):
    resp = requests.post(OVERPASS_URL, data={"data": build_query(iso)}, headers=HEADERS, timeout=150)
    resp.raise_for_status()
    return resp.json()["elements"]


def main():
    out = []
    seen_ids = set()

    for iso in PROVINCES:
        print(f"Querying Overpass for {iso}...")
        elements = fetch_province(iso)
        print(f"  {len(elements)} elements")

        for el in elements:
            key = (el["type"], el["id"])
            if key in seen_ids:
                continue
            seen_ids.add(key)

            if "center" in el:
                lat, lon = el["center"]["lat"], el["center"]["lon"]
            elif "lat" in el and "lon" in el:
                lat, lon = el["lat"], el["lon"]
            else:
                continue

            tags = el.get("tags", {})
            out.append({
                "name": tags.get("name") or "(unnamed)",
                "lat": lat,
                "lon": lon,
                "type": receptor_type(tags),
                "province": iso.split("-")[1],
                "osm_type": el["type"],
                "osm_id": el["id"],
            })

        time.sleep(2)  # be polite to the shared public instance between provinces

    out = dedupe_nearby(out)

    print(f"Total receptors: {len(out)}")
    from collections import Counter
    print("By type:", Counter(r["type"] for r in out))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
