import json,os,shutil,subprocess,sys,threading,time,uuid
from datetime import datetime,timedelta,timezone
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

ROOT=Path(__file__).resolve().parent.parent
RUNS_DIR=ROOT/'live_runs'
JOB_TIMEOUT_SECONDS=600
JOB_MAX_AGE_HOURS=48
OUTPUT_FILES=['backtraj_centerlines.geojson','backtraj_cloud.geojson','backtraj_density.geojson']

JOBS={}
JOBS_LOCK=threading.Lock()

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
