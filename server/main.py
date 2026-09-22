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
        r=subprocess.run([sys.executable,'odour/backtraj_core.py'],cwd=str(ROOT),env=env,capture_output=True,text=True,timeout=JOB_TIMEOUT_SECONDS)
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
