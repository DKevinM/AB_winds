"""
Entry point for the physics-based models (ls_model.py + gaussian_puff.py)
with the SAME interface as backtraj_core.py's __main__ block, so any
caller (run_odour_poll.sh, server/main.py, dsai/run_hrdps.py) can switch
by running this script instead:

  env in : LAT, LON, TIME_LOCAL (America/Edmonton) or TIME_UTC, HOURS,
           DIRECTION (backward|forward, default backward),
           HEIGHTS_M (default 10,40,80), OUTDIR (default odour_data),
           N_PARTICLES (default 300), Z0_M (fallback roughness, default 0.1),
           RADIUS_KM (particle travel limit, default 25; 0 = unlimited),
           STATION_WINDS (1/0, default 1: correct HRDPS toward observed
           station winds), LANDCOVER (1/0, default 1: z0 from ESA WorldCover)
  files out (unchanged names/shapes - existing pages keep working):
           backtraj_centerlines.geojson, backtraj_cloud.geojson,
           backtraj_density.geojson, backtraj_windseries.json
  files out (new):
           ls_footprint.geojson      WindTrax touchdown footprint (backward)
           gauss_footprint.geojson   Gaussian puff receptor footprint, C_R/Q_S
                                     (backward) or puff plume C/Q (forward),
                                     time-varying wind, continuous spread
           source_ranking.json       NPRI facilities inside the radius ranked
                                     by puff C_R/Q_S at their location (backward)
           model_info.json           met diagnostics (u*, L, PG, zi) + settings
"""

import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backtraj_core as bc        # noqa: E402  (MetStoreV2, GeoJSON writers)
import ls_model as ls             # noqa: E402
import gaussian_plume as gp       # noqa: E402
import gaussian_puff as gpf       # noqa: E402
import peak_to_mean as ptm        # noqa: E402
import ensemble as en             # noqa: E402
import micromet as mm             # noqa: E402
import station_obs as so          # noqa: E402
import landcover as lc            # noqa: E402
import met_store as ms            # noqa: E402
import observed_trace as ot       # noqa: E402
import drainage as dr              # noqa: E402

NPRI_PATH = "/opt/airquality/dsai_data/npri_alberta.geojson"
RANK_RADIUS_KM = 50.0


def _start_time():
    if os.environ.get("TIME_LOCAL"):
        naive = dt.datetime.fromisoformat(os.environ["TIME_LOCAL"])
        return naive.replace(tzinfo=ZoneInfo("America/Edmonton")).astimezone(dt.timezone.utc)
    if os.environ.get("TIME_UTC"):
        t = dt.datetime.fromisoformat(os.environ["TIME_UTC"].replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
    raise ValueError("Must provide TIME_LOCAL or TIME_UTC")


def _npri_sources(lat, lon, radius_km):
    try:
        with open(NPRI_PATH) as f:
            feats = json.load(f)["features"]
    except (OSError, ValueError, KeyError):
        return []
    out = []
    for ft in feats:
        p = ft.get("properties", {})
        try:
            flat, flon = float(p["Latitude"]), float(p["Longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        dx, dy = gp._local_xy(lat, lon, flat, flon)
        if math.hypot(float(dx), float(dy)) <= radius_km * 1000.0:
            out.append({"name": p.get("FacilityName"), "company": p.get("CompanyName"),
                        "npri_id": p.get("NpriID"), "sector": p.get("SectorDescriptionEn"),
                        "lat": flat, "lon": flon, "height_m": 10.0})
    return out


def _rank_from_field(field, sources, top=25):
    """Read the backward puff field (C_R/Q_S) at each facility's cell."""
    lat, lon, cq = field["lat"], field["lon"], field["cq"]
    lat_top, lon_left = float(lat[0, 0]), float(lon[0, 0])
    dlat = float(lat[1, 0] - lat[0, 0])
    dlon = float(lon[0, 1] - lon[0, 0])
    out = []
    for src in sources:
        j = int(round((src["lat"] - lat_top) / dlat))
        i = int(round((src["lon"] - lon_left) / dlon))
        if 0 <= j < cq.shape[0] and 0 <= i < cq.shape[1] and cq[j, i] > 0:
            row = dict(src)
            row["cq_s_per_m3"] = float(cq[j, i])
            dx, dy = gp._local_xy(float(lat[cq.shape[0] // 2, 0]), float(lon[0, cq.shape[1] // 2]),
                                  src["lat"], src["lon"])
            row["distance_km"] = float(math.hypot(float(dx), float(dy)) / 1000.0)
            out.append(row)
    out.sort(key=lambda r: r["cq_s_per_m3"], reverse=True)
    return out[:top]


def main():
    lat = float(os.environ["LAT"])
    lon = float(os.environ["LON"])
    start = _start_time()
    hours = float(os.environ["HOURS"])
    direction = os.environ.get("DIRECTION", "backward")
    heights_env = os.environ.get("HEIGHTS_M")
    heights = tuple(float(h) for h in heights_env.split(",")) if heights_env else (10.0, 40.0, 80.0)
    n_particles = int(os.environ.get("N_PARTICLES", "300"))
    z0 = float(os.environ.get("Z0_M", "0.1"))
    radius_km = float(os.environ.get("RADIUS_KM", "25")) or None
    use_stations = os.environ.get("STATION_WINDS", "1") != "0"
    use_landcover = os.environ.get("LANDCOVER", "1") != "0"
    outdir = Path(os.environ.get("OUTDIR", "odour_data"))
    outdir.mkdir(parents=True, exist_ok=True)

    bc.set_dem(bc.RasterDEM("data/Alberta_dem_1km.tif", "data/Alberta_slope_1km.tif"))
    # LS covers [start - hours, start] backward or [start, start + hours]
    # forward; the puff model adds a 2 h spin-up on the far side of start
    sgn = 1.0 if direction == "forward" else -1.0
    store = ms.LocalMetStore(start - dt.timedelta(hours=sgn * 2.0),
                             start + dt.timedelta(hours=sgn * hours))
    met = ls.HRDPSMet(store, z0=z0)
    notes = {}
    # valley cold-air drainage where a LiDAR-based field covers the point (Hinton so far; drainage.py)
    if os.environ.get("DRAINAGE", "1") != "0":
        try:
            dfield = dr.for_point(lat, lon)
            if dfield is not None:
                met.drainage = dfield
                notes["drainage"] = {"area": dfield.name, "S0_K": dfield.S0, "VC_kmh": round(dfield.VC * 3.6, 1),
                                     "VD_kmh": round(dfield.VD * 3.6, 1)}
        except Exception as exc:
            notes["drainage"] = {"error": str(exc)}

    if use_landcover:
        try:
            rmap = lc.RoughnessMap(lat, lon, radius_km=(radius_km or 25.0) + 5.0, month=start.month)
            met.z0_fn = rmap
            notes["landcover"] = {"source": "ESA WorldCover 2021", "season_month": start.month,
                                  "z0_at_site_m": float(rmap([lat], [lon])[0]),
                                  "class_fractions": rmap.class_fractions()}
        except Exception as exc:          # network/raster failure -> flat z0, run continues
            notes["landcover"] = {"error": str(exc), "fallback_z0_m": z0}

    obs = []
    if use_stations:
        try:
            sgn0 = 1.0 if direction == "forward" else -1.0
            t_a, t_b = sorted([start, start + dt.timedelta(hours=sgn0 * hours)])
            obs = so.fetch_station_winds(lat, lon, t_a, t_b, radius_km=(radius_km or 25.0) + 35.0)
            corr = so.StationCorrection(met, obs)
            met.correction = corr
            notes["station_winds"] = {"stations": len(corr.stations) if obs else 0, "obs_hours": len(obs),
                                      "residuals": corr.residual_summary() if obs else []}
        except Exception as exc:
            notes["station_winds"] = {"error": str(exc)}

    t0 = time.time()
    cfg = ls.LSConfig(dt_adv=60.0, radius_km=radius_km)
    res = ls.run_ls(met, lat, lon, start, hours=hours, n_particles=n_particles,
                    start_heights_m=heights, direction=direction, cfg=cfg)
    ls_secs = time.time() - t0

    def dump(name, obj):
        with open(outdir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, default=str)

    dump("backtraj_centerlines.geojson", bc.centerlines_to_geojson(res.centerlines))
    dump("backtraj_cloud.geojson", bc.cloud_to_geojson(res.cloud_points, every_n=3))
    dump("backtraj_density.geojson", bc.density_grid_to_geojson(res.cloud_points, cell_size_deg=0.01, min_count=2))

    sgn = 1.0 if direction == "forward" else -1.0
    times = [start + dt.timedelta(hours=sgn * i) for i in range(max(int(hours), 1))]

    # Wind-rose series: hourly over the model's own window (start time and
    # every hour back, or forward, to start +/- hours), from the SAME wind
    # the trajectory used - HRDPS 10 m corrected toward station winds when
    # available. Newest-in-time first (windrose.js draws index 0 solid);
    # each entry carries its time so the page can label the window instead
    # of "now / last Nh".
    tz_local = ZoneInfo("America/Edmonton")
    corrected = met.correction is not None and getattr(met.correction, "n_obs", 0) > 0
    windseries = []
    for i in range(max(int(round(hours)), 1) + 1):
        t = start + dt.timedelta(hours=sgn * i)
        try:
            s1 = met.sample(t, np.array([lat]), np.array([lon]))
            ue, vn = met.mean_wind(s1, np.array([10.0]))
        except (KeyError, ValueError):
            continue
        u, v = float(ue[0]), float(vn[0])
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        windseries.append({
            "ws": round(math.hypot(u, v), 2),
            "wd": round((math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0, 1),   # FROM
            "hours": 1,
            "time_utc": t.isoformat(),
            "time_local": t.astimezone(tz_local).strftime("%b %-d %H:%M"),
            "selected": i == 0,
            "direction": direction,
            "source": "HRDPS 10 m, station-corrected" if corrected else "HRDPS 10 m",
        })
    windseries.sort(key=lambda w: w["time_utc"], reverse=True)
    dump("backtraj_windseries.json", windseries)

    hm = gp.hourly_met(met, lat, lon, times, height_m=10.0)
    t1 = time.time()
    if direction == "forward":
        field = gpf.run_puffs(met, lat, lon, start, hours=hours, direction="forward",
                              H=min(heights), z_eval=1.5, radius_km=radius_km or 25.0)
    else:
        dump("ls_footprint.geojson", ls.touchdown_footprint_geojson(res, cell_size_deg=0.01))
        # unit source at 10 m (typical NPRI stack/vent height class) seen at a 1.5 m receptor
        field = gpf.run_puffs(met, lat, lon, start, hours=hours, direction="backward",
                              H=10.0, z_eval=1.5, radius_km=radius_km or 25.0)
        ranking = _rank_from_field(field, _npri_sources(lat, lon, radius_km or RANK_RADIUS_KM))
        # hour-mean C/Q -> perceived peaks (NSW 1 s, Ontario 10 min); see peak_to_mean.py (2026-10-21)
        pg_dom = ptm.dominant_class([h["pg"] for h in hm]) if hm else "D"
        # very stable, near-calm hours: Project Sagebrush night tests (2026-10-22) showed the model reads ground
        # concentrations ~2x LOW when u* < 0.05 m/s (beyond where surface-layer theory holds). Flag, don't tune.
        vs_hours = [h for h in hm if h["L"] > 0 and h["ustar"] < 0.05]
        if vs_hours:
            notes["very_stable_caveat"] = {
                "hours": len(vs_hours), "of": len(hm), "min_ustar": round(min(h["ustar"] for h in vs_hours), 3),
                "note": "Very stable, near-calm air (u* < 0.05 m/s): in field tests (Project Sagebrush 2016 nights) "
                        "the model read ground-level concentrations about 2x low in these conditions. Directions are "
                        "unaffected; treat levels as a lower estimate."}
        notes["peak_to_mean"] = ptm.describe(pg_dom)
        ranking = ptm.annotate(ranking, pg_dom)
        # NPRI releases -> estimated concentrations, kept in a SEPARATE file: source_ranking.json is committed
        # to the public AB_winds repo by run_odour_poll.sh, source_estimates.json is not (named-facility
        # concentration estimates are not published unless Kevin decides to). 2026-10-22
        dump("source_estimates.json", ptm.estimate_concentrations(json.loads(json.dumps(ranking))))
        # opt-in uncertainty ensemble (ENSEMBLE=<members>, e.g. 12); see ensemble.py (2026-10-21)
        n_ens = int(os.environ.get("ENSEMBLE", "0") or 0)
        if n_ens > 1:
            t_e = time.time()
            ens = en.run_ensemble(met, lat, lon, start, hours=hours, members=n_ens,
                                  n_particles=max(200, n_particles), start_heights_m=(min(heights),),
                                  sigma_dir_deg=float(os.environ.get("ENS_SIGMA_DIR", "20")),
                                  sigma_speed=float(os.environ.get("ENS_SIGMA_SPEED", "0.3")),
                                  half_width_km=min(radius_km or 25.0, 25.0))
            dump("ensemble_footprint.geojson", ens.to_geojson())
            for r in ranking:
                r["ensemble_probability"] = ens.source_probability(r["lat"], r["lon"])
                r["ensemble_sector_share"] = round(ens.sector_share(r["lat"], r["lon"]), 3)
            notes["ensemble"] = {"members": n_ens, "sigma_dir_deg": ens.sigma_dir_deg,
                                 "sigma_speed": ens.sigma_speed, "runtime_s": round(time.time() - t_e, 1),
                                 "probability": "share of members with the source bearing inside the central 80% "
                                                "of the footprint's directions at the source's distance"}
        dump("source_ranking.json", ranking)
        # observed-wind back-trace from the nearest station's own wind (2026-10-24, see observed_trace.py)
        try:
            if not obs:
                obs = so.fetch_station_winds(lat, lon, start - dt.timedelta(hours=hours), start, radius_km=ot.NEAR_KM + 1.0)
            tr = ot.back_trace(obs, lat, lon, start, hours, _npri_sources(lat, lon, radius_km or RANK_RADIUS_KM))
            if tr:
                dump("observed_trace.geojson", tr["geojson"])
                notes["observed_trace"] = tr["summary"]
            else:
                notes["observed_trace"] = {"note": f"no station within {ot.NEAR_KM:g} km with wind data"}
        except Exception as exc:
            notes["observed_trace"] = {"error": str(exc)}
    dump("gauss_footprint.geojson", gp.grid_to_geojson(field))
    puff_secs = time.time() - t1

    dump("model_info.json", {
        "model": "LS (Thomson 1987 / Flesch-Wilson-Yee 1995) + Gaussian puff",
        "direction": direction, "start_utc": start.isoformat(), "hours": hours,
        "n_particles_per_height": n_particles, "start_heights_m": heights, "z0_m": z0,
        "C0": cfg.C0, "touchdowns": len(res.touchdowns), "deposited": res.deposited,
        "radius_km": radius_km, "exited_at_radius": res.exited, "met_source": store.source,
        "meander_sigma_ms": cfg.meander_sigma, **notes,
        "ls_runtime_s": round(ls_secs, 1), "puff_runtime_s": round(puff_secs, 1),
        "gaussian": "puff, similarity spread (Taylor, z_e = H + 0.8 sigma_z), spin-up 2 h",
        "met_at_site": [{k: (v.isoformat() if k == "t" else v) for k, v in h.items()} for h in hm],
    })
    print(f"Dispersion run complete for: {lat} {lon} {start} hours: {hours} "
          f"direction: {direction} (LS {ls_secs:.1f}s, {len(res.touchdowns)} touchdowns)")


if __name__ == "__main__":
    main()
