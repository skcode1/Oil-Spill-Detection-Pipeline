#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <folder_with_tiffs> [--port 8000] [--model-load] [--no-attribution]"
  echo "Example: $0 /home/user/images --port 8000 --model-load --no-attribution"
  exit 1
fi

IMAGE_DIR="$1"
PORT="8000"
MODEL_LOAD="1"
NO_ATTR="0"

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --no-model-load)
      MODEL_LOAD="0"
      shift
      ;;
    --no-attribution)
      NO_ATTR="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Error: folder not found: $IMAGE_DIR"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -d venv ]]; then
  echo "Error: project venv not found at $PROJECT_ROOT/venv"
  echo "Create it with: python3 -m venv venv"
  exit 1
fi

source venv/bin/activate

# Pick the first TIFF in the folder as the test image.
IMAGE=$(find "$IMAGE_DIR" -type f \( -iname "*.tif" -o -iname "*.tiff" \) | head -n 1)

if [[ -z "$IMAGE" || ! -f "$IMAGE" ]]; then
  echo "Error: no TIFF image found in $IMAGE_DIR"
  exit 1
fi

echo "Project root: $PROJECT_ROOT"
echo "Testing image: $IMAGE"
echo "Backend port: $PORT"

# Start backend in background.
python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" > /tmp/oilspill_api.log 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" || true
    wait "$SERVER_PID" || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/api/v1/health" >/tmp/oilspill_health.json 2>/dev/null; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:$PORT/api/v1/health" >/tmp/oilspill_health.json 2>/dev/null; then
  echo "Error: API did not start successfully on port $PORT"
  echo "---- uvicorn log ----"
  cat /tmp/oilspill_api.log || true
  exit 1
fi

OPTIONS='{"run_attribution":false,"run_drift":true,"run_forecast":true,"drift_duration_hours":2,"save_artifacts":false,"include_preview":false}'
if [[ "$NO_ATTR" -eq 1 ]]; then
  OPTIONS='{"run_attribution":false,"run_drift":true,"run_forecast":true,"drift_duration_hours":2,"save_artifacts":false,"include_preview":false}'
fi

curl -sS -X POST "http://127.0.0.1:$PORT/api/v1/pipeline/upload" \
  -F "file=@$IMAGE" \
  -F "options=$OPTIONS" | python -m json.tool

exit 0
