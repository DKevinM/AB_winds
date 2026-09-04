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

import json
import os
import datetime as dt
from supabase import create_client

from stations import STATIONS, WATCH_STATIONS, PARAMETERS
from climatology import load_cache, cache_key, check_exceedance
from run_hysplit import run_ensemble
from fire_hotspots import check_hotspots

TRIGGERED_LOG_PATH = "/opt/airquality/dsai_data/triggered_events.json"


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


def main():
    sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    cache = load_cache()
    triggered = load_triggered_log()

    if not cache:
        print("No climatology cache found - run build_climatology_cache.py first.")
        return

    new_triggers = 0
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

            if result["any_flag"]:
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

                triggered.add(event_id)
                new_triggers += 1

    if new_triggers:
        save_triggered_log(triggered)
    print(f"Check complete. {new_triggers} new trigger(s).")


if __name__ == "__main__":
    main()
