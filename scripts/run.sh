#!/usr/bin/env bash
set -euo pipefail

# Move to repo root (works no matter where script is called from)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Activate virtual environment
if [ ! -d "venv" ]; then
  echo "[run.sh] ERROR: venv not found. Did you run 'python3 -m venv venv'?"
  exit 1
fi
source venv/bin/activate

# Load environment variables (if .env exists)
if [ -f ".env" ]; then
  set -a
  [ -f .env ] && source .env
  #source .env
  set +a
fi

# Basic sanity checks
: "${JAVAD_CHECKPOINT:?JAVAD_CHECKPOINT is not set}"
: "${S3_BUCKET:-}"   # optional, only required if UPLOAD_ENABLED=1

echo "[run.sh] Starting bird VAD pipeline"
echo "[run.sh] Device: ${DEVICE_ID:-unknown}"
echo "[run.sh] Model: ${JAVAD_MODEL_NAME:-balanced}"
echo "[run.sh] Checkpoint: ${JAVAD_CHECKPOINT}"

# Run the pipeline
# exec python -m bird_vad.pipeline

# exec PYTHONPATH=src:../javad/src python -m bird_vad.pipeline

exec env PYTHONPATH="src:../javad/src" python -m bird_vad.pipeline
