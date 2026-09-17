# dsai/watch_whitecap_incidents.py
#
# Daily watcher for recent Whitecap Resources (Saskatchewan) regulatory
# incidents that look like a real spill/leak/release - reuses this
# repo's existing HYSPLIT (GDAS archive, works for any past date)
# trajectory infrastructure, pointed at the incident's own coordinates
# instead of an Alberta AQHI station, and run FORWARD rather than
# backward (see the direction="forward" note below - this is the first
# caller to need that). That reuse is what run_hysplit.py's 2026-09-16
# `resolve_station()`/tuple support was added for - this file is the
# first (and so far only) caller that passes a (lat, lon, label) tuple
# instead of a STATIONS key.
#
# What counts as "worth a trajectory": whitecap_status_map's own
# incidents.json already carries the fields needed (added there in its
# Phase 1 review pass, 2026-09-16) - a reported spill volume > 0, H2S
# involvement, or a water-body-impact flag. Not string-matching
# INCIDENTTYPE - that field is blank on 44% of Whitecap's SK incidents.
#
# Idempotent like check_exceedances.py: a small state file tracks which
# incident IDs have already been attempted, capped at MAX_ATTEMPTS
# daily retries (HYSPLIT met data can lag a day or two to be published)
# before giving up. No separate retry-queue module at this volume - a
# handful of qualifying incidents a month, not hourly AQHI checks.
#
# HRDPS removed 2026-09-17 (Kevin's call, same day it was added): running
# both models on the first 6 real incidents surfaced genuine disagreement
# on 3 of them (up to ~100 deg apart in travel direction) - checked
# directly and it traced to real light-and-variable wind in HRDPS's own
# data for those cases (its own 10m/100m heights disagreeing with EACH
# OTHER by >100 deg in one case), not a bug in either model. HYSPLIT was
# self-consistent across heights in every case checked, matching this
# account's own pre-existing documented stance elsewhere ("HYSPLIT is
# the reliable path; HRDPS is a bonus") - so for this report specifically,
# better to show one trustworthy line than two that sometimes contradict
# each other with no way for a page viewer to know which one to believe.
# HRDPS stays exactly as-is for its original job (Alberta odour/AQHI
# receptor investigation via odour_index.py etc.) - this removal is
# Whitecap-only, this file's own concern.
#
# direction="forward" (2026-09-17 fix - Kevin caught this): every other
# caller of run_ensemble runs BACKWARD trajectories, which is the right
# physics for a receptor (an AQHI station, an odour complaint location)
# hunting for an upwind source it doesn't know yet. That's the wrong
# question here - a Whitecap incident's own site IS the known source,
# so what matters is where a release from it would travel TO, which
# needs a FORWARD trajectory instead. Reversing a backward run isn't a
# valid substitute either (wind fields vary in time and space, so a
# backward path's reverse isn't the true forward path). run_hysplit.py's
# CONTROL-file sign got a direction="forward" mode added for this - see
# its own docstring for the physics. Every other caller (check_exceedances.py,
# rerun_deep_dive.py, retry_pending*.py) keeps defaulting to backward,
# completely unaffected.

import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hysplit import run_ensemble  # noqa: E402
from parse_tdump import parse_one_tdump  # noqa: E402

WHITECAP_REPO = "/opt/airquality/github/whitecap_status_map"
INCIDENTS_PATH = f"{WHITECAP_REPO}/data/incidents.json"
OUT_DIR = f"{WHITECAP_REPO}/data/incident_trajectories"
STATE_PATH = "/opt/airquality/dsai_data/whitecap_incident_trajectories_state.json"

# 24h back, not the DSAI default of 6h - a spill's dispersion story
# matters more than "where did this come from in the last few hours"
# (that question is already answered: it came from the incident site).
TRAJ_DURATION_HOURS = 24
MAX_ATTEMPTS = 3
LOOKBACK_DAYS = 30  # a handful of days' grace past HYSPLIT met-data publication lag - see MAX_ATTEMPTS retries below

# HYSPLIT's own default start heights ([100, 500, 1000]m) were chosen
# for the Alberta BACKWARD receptor use case - bracketing plausible
# altitudes a plume might be traveling at when it reaches a station,
# since it could have been lofted well above ground. None of that
# applies to a FORWARD run starting at a ground-level Whitecap incident
# (a pipeline leak, a wellsite spill) - the release starts at the
# ground, not 500-1000m up. Kevin's call 2026-09-17: use [10, 100]m here
# instead.
WHITECAP_HEIGHTS_M = [10, 100]


def _truthy(v):
    return v in ("Yes", "Y", True, "1", 1)


def is_candidate(i):
    if i.get("status") == "Deleted":
        return False
    if i.get("lat") is None or i.get("lon") is None:
        return False
    if not i.get("occurrence_date"):
        return False
    return (
        (i.get("volume_spilled") or 0) > 0
        or _truthy(i.get("involves_h2s"))
        or _truthy(i.get("water_body_impacted"))
    )


def load_state():
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def tdump_to_geojson(results, label):
    features = []
    for height_m, tdump_path in results.items():
        if not tdump_path:
            continue
        rows = parse_one_tdump(tdump_path)
        if not rows:
            continue
        coords = [[r["lon"], r["lat"]] for r in rows]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"height_m": height_m, "label": label},
        })
    return {"type": "FeatureCollection", "features": features}


def process_incident(i, state):
    incident_id = str(i["incident_id"])
    entry = state.get(incident_id, {"attempts": 0, "hysplit_ok": False})
    if entry.get("hysplit_ok"):
        return  # fully resolved - nothing left to do
    if entry["attempts"] >= MAX_ATTEMPTS:
        return

    event_dt = dt.datetime.fromtimestamp(i["occurrence_date"] / 1000, tz=dt.timezone.utc).replace(tzinfo=None)
    label = f"WC_{incident_id}"
    coord = (i["lat"], i["lon"], label)
    os.makedirs(OUT_DIR, exist_ok=True)

    entry["attempts"] = entry.get("attempts", 0) + 1
    entry["last_tried"] = dt.datetime.now(dt.timezone.utc).isoformat()

    print(f"Incident {incident_id} @ {i['lat']:.4f},{i['lon']:.4f} occurred {event_dt.isoformat()}")

    print(f"  Running HYSPLIT ({TRAJ_DURATION_HOURS}h forward - where a release from this site goes) ...")
    try:
        results, _work_dir = run_ensemble(coord, event_dt, duration_hours=TRAJ_DURATION_HOURS, direction="forward", heights_m=WHITECAP_HEIGHTS_M)
    except Exception as e:
        print(f"  HYSPLIT failed to run at all: {e}")
        results = {}
    if any(v is not None for v in results.values()):
        geojson = tdump_to_geojson(results, label)
        out_path = f"{OUT_DIR}/{incident_id}_hysplit.geojson"
        with open(out_path, "w") as f:
            json.dump(geojson, f)
        entry["hysplit_ok"] = True
        entry["hysplit_path"] = out_path
        print(f"  HYSPLIT OK -> {out_path}")
    else:
        print(f"  HYSPLIT produced no usable trajectory (met data may not be published yet) - "
              f"will retry tomorrow (attempt {entry['attempts']}/{MAX_ATTEMPTS})")

    state[incident_id] = entry


def main():
    if not os.path.exists(INCIDENTS_PATH):
        print(f"No incidents.json found at {INCIDENTS_PATH} - nothing to do")
        return
    with open(INCIDENTS_PATH) as f:
        incidents = json.load(f)

    cutoff_ms = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000
    candidates = [i for i in incidents if is_candidate(i) and i["occurrence_date"] >= cutoff_ms]
    print(f"{len(candidates)} recent (last {LOOKBACK_DAYS}d) qualifying Whitecap incident(s) found")

    state = load_state()
    for i in candidates:
        process_incident(i, state)
    save_state(state)

    # Small index for whitecap_status_map's own dashboard build to read
    # directly, without re-deriving candidate logic - only incidents
    # with a successful trajectory are listed. Relative filenames only
    # (not this machine's absolute paths) since this file gets committed
    # into whitecap_status_map's own repo.
    index = [
        {
            "incident_id": iid,
            "hysplit_file": os.path.basename(e["hysplit_path"]) if e.get("hysplit_ok") else None,
        }
        for iid, e in state.items() if e.get("hysplit_ok")
    ]
    with open(f"{OUT_DIR}/index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"{len(index)} incident(s) with an available trajectory")


if __name__ == "__main__":
    main()
