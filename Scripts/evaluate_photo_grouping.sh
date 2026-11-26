#!/usr/bin/env bash
# Analyze a labelled photo folder with several models, compare the models,
# and calibrate a grouping policy for each one with leave-one-person-out.
#
#   Scripts/evaluate_photo_grouping.sh PHOTO_ROOT RUN_FOLDER MODELS [CALIBRATE_MODELS]
#
# MODELS is comma-separated (for example face01,w600k-r50,glintr100,lvface-b).
# CALIBRATE_MODELS defaults to MODELS; separate fused sets with spaces, for
# example "w600k-r50 glintr100 w600k-r50,glintr100".
# Requires the GPU (--device cuda). Writes everything below RUN_FOLDER.
set -euo pipefail
root=$1
run=$2
models=$3
calibrate=${4:-${models//,/ }}
python=${PYTHON:-.venv/bin/python}
logs="$run-logs"
mkdir -p "$logs"
if [ ! -f "$run/report.json" ]; then
  "$python" -m videoatlas.cli photos analyze "$root" --output "$run" --models "$models" --device cuda 2>&1 | tee "$logs/analyze.log"
fi
"$python" -m videoatlas.cli photos verify "$run" --output "$run/verify.json" 2>&1 | tee "$logs/verify.log"
for set in $calibrate; do
  # FACE01 scores sit on a lower scale than the ArcFace-family models.
  case "$set" in
    face01*) grid="0.05:0.60:0.025"; attach="none,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60" ;;
    *,*) grid="0.50:6.00:0.25"; attach="none,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0" ;;
    *) grid="0.10:0.75:0.025"; attach="none,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75" ;;
  esac
  name=${set//,/+}
  "$python" -m videoatlas.cli photos calibrate "$run" --models "$set" --merge-grid "$grid" --attach-grid "$attach" \
    --output "$run/calibration-$name.json" --policy-output "$run/policy-$name.json" 2>&1 | tee "$logs/calibrate-$name.log"
done
"$python" Scripts/summarize_photo_evaluation.py "$run" | tee "$run/summary.txt"
