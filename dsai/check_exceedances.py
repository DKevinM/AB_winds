# src/dsai/check_exceedances.py
#
# Runs hourly (cheap - reads the cached climatology, only pulls the two
# most recent readings per station/parameter). If either the absolute
# value or the hour-over-hour delta exceeds its historical percentile,
# triggers a HYSPLIT ensemble run for that station/time.
#
# Idempotent: tracks which station+timestamp combos have already
# triggered a run, so re-running within the same hour (or before new
# data lands) doesn't re-fire HYSPLIT.
#
# PM2.5 gets special handling that H2S/SO2/TRS don't: a real wildfire
# smoke episode trips the percentile threshold at most/all watched
# stations simultaneously, and keeps doing so every hour it stays
# elevated (event_id includes the timestamp, so a persisting event
# re-qualifies as "new" each hour). Left unguarded, that's a flood of
# near-duplicate HYSPLIT runs and receptor checks for what is really
# one regional event, not N point sources - exactly the failure mode
# flagged when PM2.5 was deliberately left out of PARAMETERS earlier
# (see project_dsai_hysplit memory). REGIONAL_PM25_THRESHOLD below is
# the guard: more than that many stations tripping PM2.5 in the same
# run collapses into one consolidated log line instead of N individual
# HYSPLIT+FIRMS+receptor pipelines.

import json
import os
import datetime as dt
from supabase import create_client

from stations import STATIONS, WATCH_STATIONS, PARAMETERS
from climatology import load_cache, cache_key, check_exceedance
from run_hysplit import run_ensemble
from fire_hotspots import check_hotspots
from receptors import load_receptors, receptors_downwind

TRIGGERED_LOG_PATH = "/opt/airquality/dsai_data/triggered_events.json"
PM25_PARAMETER = "Fine Particulate Matter"
REGIONAL_PM25_THRESHOLD = 5  # more than this many stations tripping PM2.5 at once = regional smoke, not N point sources


def load_triggered_log():
    if not os.path.exists(TRIGGERED_LOG_PATH):
        return set()
    with open(TRIGGERED_LOG_PATH) as f:
        return set(json.load(f))


def save_triggered_log(triggered):
    os.makedirs(os.path.dirname(TRIGGERED_LOG_PATH), exist_ok=True)
    with open(TRIGGERED_LOG_PATH, "w") as f:
        json.dump(sorted(triggered), f)


def fetch_latest_two(sb, station, parameter):
    res = (
        sb.table("aqhi_data")
        .select("ReadingDate,Value")
        .eq("StationName", station)
        .eq("ParameterName", parameter)
        .order("ReadingDate", desc=True)
        .limit(2)
        .execute()
    )
    rows = [(r["ReadingDate"], r["Value"]) for r in res.data if r["Value"] is not None]
    rows.sort(key=lambda r: r[0])  # ascending: [previous, latest]
    return rows


def fetch_latest_wind_direction(sb, station):
    res = (
        sb.table("aqhi_data")
        .select("Value")
        .eq("StationName", station)
        .eq("ParameterName", "Wind Direction")
        .order("ReadingDate", desc=True)
        .limit(1)
        .execute()
    )
    if not res.data or res.data[0]["Value"] is None:
        return None
    return float(res.data[0]["Value"])


def print_fire_context(fire):
    if fire["status"] == "ok" and fire["count"]:
        near = fire["nearest"]
        print(
            f"  FIRE CONTEXT: {fire['count']} hotspot cluster(s) within 150km - "
            f"nearest {near['distance_km']}km {near['direction']} "
            f"(FRP {near['frp']}, {near['acq_date']} {near['acq_time']} {near['daynight']})"
        )
    elif fire["status"] == "ok":
        print("  FIRE CONTEXT: no active-fire hotspots within 150km")
    elif fire["status"] == "missing":
        print("  FIRE CONTEXT: skipped (FIRMS_API_KEY not set)")
    else:
        print(f"  FIRE CONTEXT: check failed - {fire.get('error')}")


def handle_individual_trigger(sb, station, parameter, cur_ts, result, receptors):
    """Full pipeline for a single station/parameter exceedance: HYSPLIT,
    FIRMS fire context, downwind receptors. Used for every H2S/SO2/TRS
    trigger, and for PM2.5 triggers when NOT part of a regional event."""
    print(f"EXCEEDANCE: {station} / {parameter} @ {cur_ts}")
    print(f"  {result}")
    event_dt = dt.datetime.strptime(cur_ts[:16], "%Y-%m-%dT%H:%M")
    try:
        run_ensemble(station, event_dt)
        print(f"  HYSPLIT ensemble triggered for {station} @ {cur_ts}")
    except Exception as ex:
        print(f"  HYSPLIT run failed: {ex}")

    lat, lon = STATIONS[station]
    fire = check_hotspots(lat, lon)
    print_fire_context(fire)

    if receptors:
        wind_from = fetch_latest_wind_direction(sb, station)
        if wind_from is not None:
            downwind = receptors_downwind(lat, lon, wind_from, receptors, max_distance_km=30)
            if downwind:
                names = ", ".join(f"{r['name']} ({r['type']}, {r['distance_km']}km)" for r in downwind[:5])
                print(f"  DOWNWIND RECEPTORS: {len(downwind)} within 30km - {names}")
            else:
                print("  DOWNWIND RECEPTORS: none within 30km")
        else:
            print("  DOWNWIND RECEPTORS: skipped (no wind direction reading)")


def handle_regional_pm25_event(pm25_hits):
    """Collapsed handling for a regional PM2.5 event (smoke) - one FIRE
    CONTEXT check against a representative station instead of N nearly
    identical ones, no per-station HYSPLIT (a regional transport story
    told 15 times isn't 15x more useful), no per-station receptor scan
    (with this many stations affected, "what's downwind" is most of the
    territory - a per-station breakdown isn't the useful cut for a
    genuinely regional event)."""
    stations_hit = [s for s, _, _ in pm25_hits]
    print(f"REGIONAL SMOKE EVENT: PM2.5 exceedance at {len(stations_hit)} stations simultaneously - {', '.join(stations_hit)}")
    print(f"  Skipping {len(stations_hit)} individual HYSPLIT/receptor runs - one regional event, not {len(stations_hit)} point sources.")

    rep_station = stations_hit[0]
    lat, lon = STATIONS[rep_station]
    fire = check_hotspots(lat, lon)
    print(f"  (representative check from {rep_station})")
    print_fire_context(fire)


def main():
    sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    cache = load_cache()
    triggered = load_triggered_log()
    receptors = load_receptors()

    if not cache:
        print("No climatology cache found - run build_climatology_cache.py first.")
        return

    new_triggers = 0
    pm25_hits = []  # (station, cur_ts, event_id) - decided after the full sweep

    for station in WATCH_STATIONS:
        for parameter in PARAMETERS:
            key = cache_key(station, parameter)
            clim = cache.get(key)
            if not clim:
                continue

            latest_two = fetch_latest_two(sb, station, parameter)
            if len(latest_two) < 2:
                continue

            (prev_ts, prev_val), (cur_ts, cur_val) = latest_two
            event_id = f"{station}::{parameter}::{cur_ts}"
            if event_id in triggered:
                continue

            delta = cur_val - prev_val
            result = check_exceedance(clim, cur_val, delta)

            if not result["any_flag"]:
                continue

            if parameter == PM25_PARAMETER:
                pm25_hits.append((station, cur_ts, event_id))
            else:
                handle_individual_trigger(sb, station, parameter, cur_ts, result, receptors)
                triggered.add(event_id)
                new_triggers += 1

    if pm25_hits:
        if len(pm25_hits) > REGIONAL_PM25_THRESHOLD:
            handle_regional_pm25_event(pm25_hits)
            for _, _, event_id in pm25_hits:
                triggered.add(event_id)
                new_triggers += 1
        else:
            for station, cur_ts, event_id in pm25_hits:
                key = cache_key(station, PM25_PARAMETER)
                clim = cache[key]
                latest_two = fetch_latest_two(sb, station, PM25_PARAMETER)
                (prev_ts, prev_val), (_, cur_val) = latest_two
                result = check_exceedance(clim, cur_val, cur_val - prev_val)
                handle_individual_trigger(sb, station, PM25_PARAMETER, cur_ts, result, receptors)
                triggered.add(event_id)
                new_triggers += 1

    if new_triggers:
        save_triggered_log(triggered)
    print(f"Check complete. {new_triggers} new trigger(s).")


if __name__ == "__main__":
    main()
