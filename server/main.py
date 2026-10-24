import json,os,shutil,subprocess,sys,threading,time,uuid
from datetime import datetime,timedelta,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

SASKATCHEWAN_TZ=ZoneInfo('America/Regina')

ROOT=Path(__file__).resolve().parent.parent
RUNS_DIR=ROOT/'live_runs'
JOB_TIMEOUT_SECONDS=600
JOB_MAX_AGE_HOURS=48
OUTPUT_FILES=['backtraj_centerlines.geojson','backtraj_cloud.geojson','backtraj_density.geojson']

JOBS={}
JOBS_LOCK=threading.Lock()

# Whitecap status map's "click anywhere, run a forward HYSPLIT
# trajectory" feature (added 2026-09-22) - separate job dict/prefix so
# its job_ids can never collide with the odour tool's, but the same
# submit/poll pattern and worker thread, just running a different
# script (dsai/run_click_trajectory.py - real HYSPLIT via
# run_hysplit.run_ensemble, same engine already used for Whitecap's
# real incident trajectories, deliberately NOT odour/backtraj_core.py's
# different particle-cloud model - Kevin's call, explicit: "12hrs at
# 10m and 100m as usual"). cwd is dsai/, not ROOT, because that
# script's own imports (run_hysplit, watch_whitecap_incidents) assume
# dsai/ is the working directory - same convention
# run_dsai_watch_whitecap.sh already uses to invoke that sibling script.
CLICK_RUNS_DIR=ROOT/'click_runs'
CLICK_JOB_TIMEOUT_SECONDS=600
CLICK_JOBS={}
CLICK_JOBS_LOCK=threading.Lock()

def _run_click_job(job_id,lat,lon,time_local):
    with CLICK_JOBS_LOCK:CLICK_JOBS[job_id]['status']='running'
    outdir=CLICK_RUNS_DIR/job_id
    outdir.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update({'LAT':str(lat),'LON':str(lon),'TIME_LOCAL':time_local,'OUTDIR':str(outdir)})
    try:
        r=subprocess.run([sys.executable,'run_click_trajectory.py'],cwd=str(ROOT/'dsai'),env=env,capture_output=True,text=True,timeout=CLICK_JOB_TIMEOUT_SECONDS)
        error_path=outdir/'error.json'
        traj_path=outdir/'trajectory.geojson'
        if error_path.exists():
            with CLICK_JOBS_LOCK:CLICK_JOBS[job_id].update(status='failed',error=json.loads(error_path.read_text()).get('error','HYSPLIT run failed'))
            return
        if r.returncode!=0 or not traj_path.exists():
            with CLICK_JOBS_LOCK:CLICK_JOBS[job_id].update(status='failed',error=(r.stderr or r.stdout)[-4000:] or 'No trajectory output produced.')
            return
        with CLICK_JOBS_LOCK:CLICK_JOBS[job_id].update(status='completed',result=json.loads(traj_path.read_text()),completed_at=datetime.now(timezone.utc).isoformat())
    except subprocess.TimeoutExpired:
        with CLICK_JOBS_LOCK:CLICK_JOBS[job_id].update(status='failed',error=f'Model run exceeded {CLICK_JOB_TIMEOUT_SECONDS}s timeout.')
    except Exception as ex:
        with CLICK_JOBS_LOCK:CLICK_JOBS[job_id].update(status='failed',error=f'{type(ex).__name__}: {ex}')

app=FastAPI(title='AB_winds Back-Trajectory (live)')
INDEX_HTML=(Path(__file__).parent/'templates'/'index.html').read_text(encoding='utf-8')
CARTO_API_KEY=os.environ.get('CARTO_API_KEY','')
if CARTO_API_KEY:
    _carto_layer_js=f"var cartoLayer=L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/dark_all/{{z}}/{{x}}/{{y}}.png?key={CARTO_API_KEY}',{{attribution:'&copy; OpenStreetMap contributors &copy; CARTO',subdomains:'abcd',maxZoom:20}});"
    _carto_base_layers="{'Light':osmLayer,'Dark':cartoLayer}"
else:
    _carto_layer_js=''
    _carto_base_layers="{'Light':osmLayer}"
INDEX_HTML=INDEX_HTML.replace('__CARTO_DARK_LAYER_JS__',_carto_layer_js).replace('__CARTO_BASE_LAYERS__',_carto_base_layers)

def _cleanup_old_runs():
    if not RUNS_DIR.exists():return
    cutoff=time.time()-JOB_MAX_AGE_HOURS*3600
    for d in RUNS_DIR.iterdir():
        if d.is_dir() and d.stat().st_mtime<cutoff:
            shutil.rmtree(d,ignore_errors=True)

def _run_job(job_id,lat,lon,time_local,hours):
    with JOBS_LOCK:JOBS[job_id]['status']='running'
    outdir=RUNS_DIR/job_id
    outdir.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update({'LAT':str(lat),'LON':str(lon),'TIME_LOCAL':time_local,'HOURS':str(hours),'OUTDIR':str(outdir)})
    try:
        # 2026-10-13: run_dispersion.py (LS + Gaussian puff) replaces backtraj_core.py; same output files
        r=subprocess.run([sys.executable,'odour/run_dispersion.py'],cwd=str(ROOT),env=env,capture_output=True,text=True,timeout=JOB_TIMEOUT_SECONDS)
        if r.returncode!=0:
            with JOBS_LOCK:JOBS[job_id].update(status='failed',error=(r.stderr or r.stdout)[-4000:])
            return
        result={}
        for fname in OUTPUT_FILES:
            fpath=outdir/fname
            if fpath.exists():
                result[fname.replace('.geojson','')]=json.loads(fpath.read_text())
        with JOBS_LOCK:JOBS[job_id].update(status='completed',result=result,completed_at=datetime.now(timezone.utc).isoformat())
    except subprocess.TimeoutExpired:
        with JOBS_LOCK:JOBS[job_id].update(status='failed',error=f'Model run exceeded {JOB_TIMEOUT_SECONDS}s timeout.')
    except Exception as ex:
        with JOBS_LOCK:JOBS[job_id].update(status='failed',error=f'{type(ex).__name__}: {ex}')

@app.get('/',response_class=HTMLResponse)
def index():
    return INDEX_HTML

@app.post('/run')
def run(lat:float=Query(...,ge=-90,le=90),lon:float=Query(...,ge=-180,le=180),time_local:str=Query(...),hours:float=Query(6,ge=1,le=48)):
    try:
        datetime.fromisoformat(time_local)
    except ValueError:
        return JSONResponse({'error':'time_local must be an ISO datetime, e.g. 2026-07-20T14:00:00'},status_code=400)
    _cleanup_old_runs()
    job_id=uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id]={'status':'pending','submitted_at':datetime.now(timezone.utc).isoformat(),'lat':lat,'lon':lon,'time_local':time_local,'hours':hours}
    threading.Thread(target=_run_job,args=(job_id,lat,lon,time_local,hours),daemon=True).start()
    return {'job_id':job_id,'status':'pending'}

@app.get('/status/{job_id}')
def status(job_id:str):
    with JOBS_LOCK:
        job=JOBS.get(job_id)
    if not job:return JSONResponse({'status':'not_found'},status_code=404)
    return job

def _cleanup_old_click_runs():
    if not CLICK_RUNS_DIR.exists():return
    cutoff=time.time()-JOB_MAX_AGE_HOURS*3600
    for d in CLICK_RUNS_DIR.iterdir():
        if d.is_dir() and d.stat().st_mtime<cutoff:
            shutil.rmtree(d,ignore_errors=True)

@app.post('/run-click')
def run_click(lat:float=Query(...,ge=-90,le=90),lon:float=Query(...,ge=-180,le=180),time_local:str=Query(...)):
    try:
        parsed=datetime.fromisoformat(time_local)
    except ValueError:
        return JSONResponse({'error':'time_local must be an ISO datetime, e.g. 2026-09-22T14:00:00'},status_code=400)
    # Real HYSPLIT met-data coverage window, not an arbitrary guess -
    # near-real-time gfsa cycles roll off after about a week (see
    # gdas_fetch.py's own module docstring), and a future start time
    # obviously has no data at all yet. time_local is Saskatchewan local
    # (same convention as run_click_trajectory.py itself), so it must be
    # converted to UTC before comparing against a UTC "now" - comparing
    # the naive values directly was off by the -6h SK/UTC offset (e.g. a
    # time_local that's actually still 6h in the future would pass).
    now=datetime.now(timezone.utc)
    parsed_utc=parsed.replace(tzinfo=SASKATCHEWAN_TZ).astimezone(timezone.utc)
    if parsed_utc>now:
        return JSONResponse({'error':'Start time cannot be in the future.'},status_code=400)
    if (now-parsed_utc).days>6:
        return JSONResponse({'error':'Start time is more than 6 days ago - near-real-time met data may no longer cover it.'},status_code=400)
    _cleanup_old_click_runs()
    job_id=uuid.uuid4().hex[:12]
    with CLICK_JOBS_LOCK:
        CLICK_JOBS[job_id]={'status':'pending','submitted_at':datetime.now(timezone.utc).isoformat(),'lat':lat,'lon':lon,'time_local':time_local}
    threading.Thread(target=_run_click_job,args=(job_id,lat,lon,time_local),daemon=True).start()
    return {'job_id':job_id,'status':'pending'}

@app.get('/click-status/{job_id}')
def click_status(job_id:str):
    with CLICK_JOBS_LOCK:
        job=CLICK_JOBS.get(job_id)
    if not job:return JSONResponse({'status':'not_found'},status_code=404)
    return job


# ---------------------------------------------------------------------------
# WCAS emergency page: LS + Gaussian puff dispersion, forward OR backward
# (added 2026-10-15). Proxied as /wcas-emergency/api/ behind that page's
# password. Same run_dispersion.py as /run, but with the direction exposed
# and the model's limits enforced HERE, server-side, not just in the page:
#   - 1..12 h (runtime ~25 s forward / ~75 s backward per 6 h)
#   - the whole run must sit inside the HRDPS files we actually have
#     (backward: start - hours >= earliest file; forward: start + hours <=
#     latest forecast valid time) - otherwise particles run on no wind
#   - particles stop at RADIUS_KM (25 km, the model's built-in travel limit)
#   - at most LS_MAX_RUNNING runs at once (CPU), else 429
# Returns only what the map draws: centre lines, the Gaussian puff footprint
# trimmed to the cells holding 99% of the weight, the LS touchdown footprint
# and NPRI source ranking (backward), and the met summary.
import re as _re
sys.path.insert(0, str(ROOT / 'py'))
import wind_archive  # noqa: E402

LS_RUNS_DIR = ROOT / 'ls_runs'
LS_MAX_HOURS = 12
LS_RADIUS_KM = 25
LS_MAX_RUNNING = 2
LS_JOBS = {}
LS_JOBS_LOCK = threading.Lock()
EDMONTON_TZ = ZoneInfo('America/Edmonton')
_MET_RE = _re.compile(r'ab_met_(\d{8})_(\d{2})Z_f(\d{3})\.json\.gz$')


def _met_coverage():
    """(earliest, latest) valid time (UTC) across local HRDPS files."""
    times = []
    for f in (ROOT / 'met_data').glob('ab_met_*.json.gz'):
        m = _MET_RE.search(f.name)
        if m:
            run = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H').replace(tzinfo=timezone.utc)
            times.append(run + timedelta(hours=int(m.group(3))))
    # Older months are archived in Dropbox and restored on demand by met_store.
    archived = wind_archive.archived_first_valid()
    if archived:
        times.append(archived)
    return (min(times), max(times)) if times else (None, None)


def _trim_footprint(fc, keep=0.99):
    feats = sorted(fc.get('features', []), key=lambda f: -(f['properties'].get('weight') or 0))
    tot = sum(f['properties'].get('weight') or 0 for f in feats) or 1.0
    out, acc = [], 0.0
    for f in feats:
        out.append(f); acc += f['properties'].get('weight') or 0
        if acc / tot >= keep:
            break
    return _round_fc({'type': 'FeatureCollection', 'features': out})


def _round_fc(fc, nd=5):
    """Coordinates to ~1 m, numbers to 4 significant figures - keeps the browser payload small."""
    if not fc:
        return fc
    rc = lambda c: [round(c[0], nd), round(c[1], nd)] if isinstance(c[0], (int, float)) else [rc(x) for x in c]
    for f in fc.get('features', []):
        g = f.get('geometry') or {}
        if 'coordinates' in g:
            g['coordinates'] = rc(g['coordinates'])
        f['properties'] = {k: (float(f'{v:.4g}') if isinstance(v, float) else v) for k, v in (f.get('properties') or {}).items()}
    return fc


def _run_ls_job(job_id, lat, lon, time_local, hours, direction):
    with LS_JOBS_LOCK: LS_JOBS[job_id]['status'] = 'running'
    outdir = LS_RUNS_DIR / job_id
    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({'LAT': str(lat), 'LON': str(lon), 'TIME_LOCAL': time_local, 'HOURS': str(hours),
                'DIRECTION': direction, 'RADIUS_KM': str(LS_RADIUS_KM), 'OUTDIR': str(outdir)})
    try:
        r = subprocess.run([sys.executable, 'odour/run_dispersion.py'], cwd=str(ROOT), env=env,
                           capture_output=True, text=True, timeout=JOB_TIMEOUT_SECONDS)
        if r.returncode != 0:
            with LS_JOBS_LOCK: LS_JOBS[job_id].update(status='failed', error=(r.stderr or r.stdout)[-2000:])
            return
        rd = lambda n: json.loads((outdir / n).read_text()) if (outdir / n).exists() else None
        info = rd('model_info.json') or {}
        rank = rd('source_ranking.json')
        result = {
            'centerlines': _round_fc(rd('backtraj_centerlines.geojson')),
            'puff_footprint': _trim_footprint(rd('gauss_footprint.geojson') or {}),
            'ls_footprint': _round_fc(rd('ls_footprint.geojson')) if direction == 'backward' else None,
            'ls_density': _round_fc(rd('backtraj_density.geojson')),   # where the LS particles were (both directions)
            'source_ranking': (rank or [])[:10] if isinstance(rank, list) else None,
            'observed_trace': rd('observed_trace.geojson'),   # path from the nearest station's own wind (2026-10-24)
            # NPRI release-based concentration estimates (2026-10-22). /ls-run is only reachable through the
            # password-protected pages (nginx: /wcas-emergency/api, /hinton/api; uvicorn binds 127.0.0.1).
            'source_estimates': (rd('source_estimates.json') or [])[:10],
            'info': {k: info.get(k) for k in ('direction', 'start_utc', 'hours', 'radius_km', 'exited_at_radius',
                                              'n_particles_per_height', 'start_heights_m',
                                              'touchdowns', 'met_at_site', 'station_winds', 'landcover', 'met_source',
                                              'very_stable_caveat', 'peak_to_mean', 'observed_trace', 'drainage')},
        }
        with LS_JOBS_LOCK: LS_JOBS[job_id].update(status='completed', result=result, completed_at=datetime.now(timezone.utc).isoformat())
    except subprocess.TimeoutExpired:
        with LS_JOBS_LOCK: LS_JOBS[job_id].update(status='failed', error=f'Model run exceeded {JOB_TIMEOUT_SECONDS}s.')
    except Exception as ex:
        with LS_JOBS_LOCK: LS_JOBS[job_id].update(status='failed', error=f'{type(ex).__name__}: {ex}')
    finally:
        shutil.rmtree(outdir, ignore_errors=True)   # result is held in memory; nothing to keep on disk


@app.get('/ls-limits')
def ls_limits():
    first, last = _met_coverage()
    return {'max_hours': LS_MAX_HOURS, 'radius_km': LS_RADIUS_KM,
            'met_first_utc': first.isoformat() if first else None, 'met_last_utc': last.isoformat() if last else None}


@app.post('/ls-run')
def ls_run(lat: float = Query(..., ge=48, le=62), lon: float = Query(..., ge=-134, le=-88), time_local: str = Query(...),
           hours: float = Query(6, ge=1, le=LS_MAX_HOURS), direction: str = Query('backward', pattern='^(forward|backward)$')):
    try:
        start = datetime.fromisoformat(time_local).replace(tzinfo=EDMONTON_TZ).astimezone(timezone.utc)
    except ValueError:
        return JSONResponse({'error': 'time_local must be like 2026-10-15T14:00:00 (Alberta time)'}, status_code=400)
    first, last = _met_coverage()
    if first is None:
        return JSONResponse({'error': 'No wind data on the server.'}, status_code=503)
    span = timedelta(hours=hours)
    lo, hi = (start - span, start) if direction == 'backward' else (start, start + span)
    if lo < first:
        return JSONResponse({'error': f'Wind data only starts {first.astimezone(EDMONTON_TZ):%b %d %H:%M} - pick a later time or fewer hours.'}, status_code=400)
    if hi > last:
        return JSONResponse({'error': f'The wind forecast only reaches {last.astimezone(EDMONTON_TZ):%b %d %H:%M} - pick an earlier time or fewer hours.'}, status_code=400)
    with LS_JOBS_LOCK:
        # forget finished runs after an hour; cap concurrent runs
        for k in [k for k, j in LS_JOBS.items() if j['status'] in ('completed', 'failed')
                  and datetime.fromisoformat(j['submitted_at']) < datetime.now(timezone.utc) - timedelta(hours=1)]:
            LS_JOBS.pop(k, None)
        if sum(j['status'] in ('pending', 'running') for j in LS_JOBS.values()) >= LS_MAX_RUNNING:
            return JSONResponse({'error': 'The model is busy with other runs - try again in a minute.'}, status_code=429)
        job_id = uuid.uuid4().hex[:12]
        LS_JOBS[job_id] = {'status': 'pending', 'submitted_at': datetime.now(timezone.utc).isoformat(), 'lat': lat, 'lon': lon,
                           'time_local': time_local, 'hours': hours, 'direction': direction}
    threading.Thread(target=_run_ls_job, args=(job_id, lat, lon, time_local, hours, direction), daemon=True).start()
    return {'job_id': job_id, 'status': 'pending'}


@app.get('/ls-status/{job_id}')
def ls_status(job_id: str):
    with LS_JOBS_LOCK:
        job = LS_JOBS.get(job_id)
    if not job: return JSONResponse({'status': 'not_found'}, status_code=404)
    return job
