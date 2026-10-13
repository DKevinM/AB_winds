#!/bin/bash
set -e

cd /opt/airquality/github/AB_winds
source /opt/airquality/venv/bin/activate
set -a
source /opt/airquality/config/intelligence.env
set +a

LOCKFILE="/opt/airquality/locks/ab_winds_git.lock"
mkdir -p "$(dirname "$LOCKFILE")"

(
  flock -w 120 200
  git fetch origin
  git pull --rebase origin main
) 200>"$LOCKFILE"

if [ ! -f trigger_request.json ]; then
  echo "No trigger_request.json - nothing to do."
  exit 0
fi

STATUS=$(python3 -c "import json; print(json.load(open('trigger_request.json')).get('status',''))")

if [ "$STATUS" != "pending" ]; then
  echo "trigger_request.json status is '$STATUS' - nothing to do."
  exit 0
fi

echo "Pending request found - running odour back trajectory model."

export LAT=$(python3 -c "import json; print(json.load(open('trigger_request.json'))['lat'])")
export LON=$(python3 -c "import json; print(json.load(open('trigger_request.json'))['lon'])")
export TIME_LOCAL=$(python3 -c "import json; print(json.load(open('trigger_request.json'))['time_local'])")
export HOURS=$(python3 -c "import json; print(json.load(open('trigger_request.json'))['hours'])")

# 2026-10-13: physics-based LS + Gaussian puff model (odour/run_dispersion.py)
# replaces odour/backtraj_core.py - same inputs and the same three map files,
# plus footprint maps. Roll back by pointing this line at backtraj_core.py.
if python3 odour/run_dispersion.py; then
  python3 - <<'PYEOF'
import json, datetime
req = json.load(open("trigger_request.json"))
req["status"] = "completed"
req["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
json.dump(req, open("trigger_request.json", "w"), indent=2)
PYEOF
  echo "Model run complete."
else
  python3 - <<'PYEOF'
import json, datetime
req = json.load(open("trigger_request.json"))
req["status"] = "failed"
req["failed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
json.dump(req, open("trigger_request.json", "w"), indent=2)
PYEOF
  echo "Model run FAILED."
fi

(
  flock -w 120 200

  git add odour_data/*.geojson trigger_request.json
  # Conditional: a run that fails before this write step (or an older run)
  # won't have the file, and set -e would otherwise abort this whole block
  # (including the trigger_request.json status push) on git add's missing-
  # pathspec error.
  if [ -f odour_data/backtraj_windseries.json ]; then
    git add odour_data/backtraj_windseries.json
  fi
  for f in odour_data/source_ranking.json odour_data/model_info.json; do
    if [ -f "$f" ]; then git add "$f"; fi
  done

  if git diff --cached --quiet; then
      echo "No changes to commit."
      exit 0
  fi

  git commit -m "Odour trajectory run: $(python3 -c "import json; print(json.load(open('trigger_request.json'))['status'])")"
  for attempt in 1 2 3; do
      if git push origin main; then
          break
      fi
      echo "push rejected (attempt $attempt/3); rebasing onto latest and retrying..."
      git pull --rebase origin main
  done
) 200>"$LOCKFILE"
