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
from climatology import load_cache, cache_key, check_exceedance, is_extreme
from run_hysplit import run_ensemble, DEFAULT_DURATION_HOURS
from fire_hotspots import check_hotspots
from receptors import load_receptors, receptors_downwind
import trajectory_retry_queue
import trs_retry_queue
import dual_model_queue

TRIGGERED_LOG_PATH = "/opt/airquality/dsai_data/triggered_events.json"
PM25_PARAMETER = "Fine Particulate Matter"
REGIONAL_PM25_THRESHOLD = 5  # more than this many stations tripping PM2.5 at once = regional smoke, not N point sources
TRS_PARAMETER = "Total Reduced Sulphur"
# Automatic HYSPLIT on every trigger PAUSED 2026-10-23 (Kevin's call): most
# same-hour runs failed on not-yet-published GFS met, the retry backlog grew
# into the thousands, and no page displays these trajectories. Detection,
# fire context and receptors still run and log; trajectories are now run on
# request (rerun_deep_dive.py / run_hysplit.py). Set DSAI_AUTO_HYSPLIT=1 to restore.
AUTO_HYSPLIT = os.getenv("DSAI_AUTO_HYSPLIT", "0") == "1"

# H2S/TRS/SO2 trigger rules (2026-10-23, Kevin's call). The percentile test
# alone fired ~2,300 times in 30 days at a median of 1-2 ppb - these analysers
# report whole ppb and sit near 0, so a single 1 ppb reading ranks as
# "unusual" (Smoky Heights TRS: 499 triggers, every one at 1.0). Now:
#   - ignore anything under the parameter's floor, whatever its percentile
#   - always trigger at the lowest 1-h alert level, percentile or not
#   - tag 24-h means at/over the 24-h objective
#   - back-to-back hours are one episode: full checks on the first hour,
#     one CONTINUING line after that
# TRS has no Alberta objective of its own; the H2S levels are used as
# reference. All values ppb.
H2S_PARAMETER = "Hydrogen Sulphide"
SO2_PARAMETER = "Sulphur Dioxide"
SULPHUR_RULES = {
    H2S_PARAMETER: {
        "floor": 3.0,
        "alerts_1h": [(10.0, "Alberta 1-h objective")],
        "alert_24h": (3.0, "Alberta 24-h objective"),
    },
    TRS_PARAMETER: {
        "floor": 3.0,
        "alerts_1h": [(10.0, "Alberta H2S 1-h objective, as reference")],
        "alert_24h": None,
    },
    SO2_PARAMETER: {
        "floor": 10.0,  # 5 ppb gave ~100 episodes/30 d, mostly Heartland background; 10 gave 29
        "alerts_1h": [(65.0, "2025 CAAQS 1-h level, as reference"), (172.0, "Alberta 1-h objective")],
        "alert_24h": (48.0, "Alberta 24-h objective"),
    },
}
SULPHUR_PARAMETERS = tuple(SULPHUR_RULES)
EPISODE_GAP_HOURS = 2         # up to one quiet/missing hour keeps the same episode
EPISODES_PATH = "/opt/airquality/dsai_data/sulphur_episodes.json"


def _enqueue_retry(station, parameter, event_dt, duration_hours, cur_ts, result=None):
    """Routes a failed HYSPLIT run to the right retry queue - TRS gets
    parked for a monthly as-needed check (Kevin's call 2026-09-15, see
    trs_retry_queue.py), everything else keeps the daily 01:00 cadence."""
    if parameter == TRS_PARAMETER:
        trs_retry_queue.enqueue(station, parameter, event_dt, duration_hours, cur_ts, result=result)
    else:
        trajectory_retry_queue.enqueue(station, parameter, event_dt, duration_hours, cur_ts, result=result)


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


def fetch_24h_mean(sb, station, parameter, cur_ts):
    start = (dt.datetime.fromisoformat(cur_ts) - dt.timedelta(hours=23)).isoformat()
    res = (
        sb.table("aqhi_data")
        .select("Value")
        .eq("StationName", station)
        .eq("ParameterName", parameter)
        .gte("ReadingDate", start)
        .lte("ReadingDate", cur_ts)
        .execute()
    )
    vals = [r["Value"] for r in res.data if r["Value"] is not None]
    if len(vals) < 18:  # need 75% of the day for a meaningful 24-h mean
        return None
    return sum(vals) / len(vals)


def sulphur_alert_notes(sb, station, parameter, cur_ts, cur_val):
    rules = SULPHUR_RULES[parameter]
    notes = []
    for level, label in rules["alerts_1h"]:
        if cur_val >= level:
            notes.append(f"ALERT: {cur_val:g} ppb >= {level:g} ppb {label}")
    if rules["alert_24h"]:
        level, label = rules["alert_24h"]
        mean24 = fetch_24h_mean(sb, station, parameter, cur_ts)
        if mean24 is not None and mean24 >= level:
            notes.append(f"ALERT: 24-h mean {mean24:.1f} ppb >= {level:g} ppb {label}")
    return notes


def sulphur_triggers(parameter, cur_val, result):
    rules = SULPHUR_RULES[parameter]
    if cur_val >= min(level for level, _ in rules["alerts_1h"]):
        return True
    return cur_val >= rules["floor"] and result["any_flag"]


def load_episodes():
    if not os.path.exists(EPISODES_PATH):
        return {}
    with open(EPISODES_PATH) as f:
        return json.load(f)


def save_episodes(episodes):
    with open(EPISODES_PATH, "w") as f:
        json.dump(episodes, f, indent=1, sort_keys=True)


def update_episode(episodes, station, parameter, cur_ts, cur_val):
    """Returns (episode, is_new). Extends the station's open episode if its
    last hour is within EPISODE_GAP_HOURS, otherwise starts a new one."""
    key = f"{station}::{parameter}"
    now = dt.datetime.fromisoformat(cur_ts)
    ep = episodes.get(key)
    if ep and dt.timedelta(0) < now - dt.datetime.fromisoformat(ep["last"]) <= dt.timedelta(hours=EPISODE_GAP_HOURS):
        ep["last"] = cur_ts
        ep["hours"] += 1
        ep["peak"] = max(ep["peak"], cur_val)
        return ep, False
    ep = {"start": cur_ts, "last": cur_ts, "hours": 1, "peak": cur_val, "alerted": False}
    episodes[key] = ep
    return ep, True


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


def handle_individual_trigger(sb, station, parameter, cur_ts, result, receptors, notes=()):
    """Full pipeline for a single station/parameter exceedance: HYSPLIT,
    FIRMS fire context, downwind receptors. Used for the first hour of an
    H2S/TRS/SO2 episode, and for PM2.5 triggers when NOT part of a
    regional event."""
    print(f"EXCEEDANCE: {station} / {parameter} @ {cur_ts}")
    print(f"  {result}")
    for note in notes:
        print(f"  {note}")
    event_dt = dt.datetime.strptime(cur_ts[:16], "%Y-%m-%dT%H:%M")
    if not AUTO_HYSPLIT:
        print("  HYSPLIT: automatic runs paused - run on request")
    else:
        _run_trajectory(station, parameter, event_dt, cur_ts, result)

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


def _run_trajectory(station, parameter, event_dt, cur_ts, result):
    try:
        results, work_dir = run_ensemble(station, event_dt)
        if any(v is not None for v in results.values()):
            print(f"  HYSPLIT ensemble triggered for {station} @ {cur_ts}")
            if is_extreme(result):
                # Real HRDPS runs take ~7min (measured 2026-09-13) - too
                # slow to run here without risking this hourly check
                # overrunning into the next one. Queued for
                # run_dsai_dual_model.sh instead (every 20min) - see
                # dual_model_queue.py.
                print(f"  Extreme reading - queued for dual-model (HYSPLIT vs HRDPS) comparison")
                dual_model_queue.enqueue(station, parameter, event_dt, DEFAULT_DURATION_HOURS, cur_ts, work_dir)
        else:
            # Every height failed - not necessarily a real problem, often
            # just the met data for this hour not being published yet
            # (see run_hysplit.py's 2026-09-13 fix). Queue it rather than
            # lose the trigger: retried at 01:00 tomorrow and, if still
            # short, once more the day after - see trajectory_retry_queue.py.
            print(f"  HYSPLIT run produced no usable trajectory (met data likely not published yet) - queued for delayed retry")
            _enqueue_retry(station, parameter, event_dt, DEFAULT_DURATION_HOURS, cur_ts, result=result)
    except Exception as ex:
        print(f"  HYSPLIT run failed: {ex}")
        _enqueue_retry(station, parameter, event_dt, DEFAULT_DURATION_HOURS, cur_ts, result=result)


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
    episodes = load_episodes()
    episodes_changed = False

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

            if parameter in SULPHUR_PARAMETERS:
                if not sulphur_triggers(parameter, cur_val, result):
                    continue
                notes = sulphur_alert_notes(sb, station, parameter, cur_ts, cur_val)
                ep, is_new = update_episode(episodes, station, parameter, cur_ts, cur_val)
                episodes_changed = True
                if is_new:
                    handle_individual_trigger(sb, station, parameter, cur_ts, result, receptors, notes)
                else:
                    print(f"CONTINUING: {station} / {parameter} @ {cur_ts} - {cur_val:g} ppb, "
                          f"hour {ep['hours']} of episode since {ep['start']} (peak {ep['peak']:g} ppb)")
                    if notes and not ep["alerted"]:
                        for note in notes:
                            print(f"  {note}")
                ep["alerted"] = ep["alerted"] or bool(notes)
                triggered.add(event_id)
                new_triggers += 1
                continue

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
    if episodes_changed:
        save_episodes(episodes)
    print(f"Check complete. {new_triggers} new trigger(s).")


if __name__ == "__main__":
    main()
