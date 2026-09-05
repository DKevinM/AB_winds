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

    print(f"Total receptors: {len(out)}")
    from collections import Counter
    print("By type:", Counter(r["type"] for r in out))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
