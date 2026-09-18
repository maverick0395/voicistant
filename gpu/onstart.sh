#!/usr/bin/env bash
# GPU box bootstrap. Runs on every vast.ai instance start (template on-start, backgrounded).
# Idempotent: first boot installs and downloads everything, later boots reuse the disk.
# Order matters: vLLM reserves its memory share first, speaches fits into the rest.
set -euo pipefail

: "${MODEL_API_KEY:?}" "${VPS_HOST:?}" "${VPS_HOST_KEY:?}" "${TUNNEL_KEY_B64:?}"
LLM_MODEL=${LLM_MODEL:-QuantTrio/Qwen3.5-9B-AWQ}
LLM_TOOL_PARSER=${LLM_TOOL_PARSER:-qwen3_coder}     # Qwen3 (fallback model): hermes
LLM_EXTRA_ARGS=${LLM_EXTRA_ARGS:---language-model-only}  # Qwen3 (fallback model): empty
STT_MODEL=${STT_MODEL:-deepdml/faster-whisper-large-v3-turbo-ct2}
TTS_MODEL=${TTS_MODEL:-speaches-ai/Kokoro-82M-v1.0-ONNX}
SPEACHES_REF=${SPEACHES_REF:-v0.9.0-rc.3}
VPS_SSH_PORT=${VPS_SSH_PORT:-22}
IDLE_STOP_MINUTES=${IDLE_STOP_MINUTES:-30}

LOG=/var/log/voicistant
mkdir -p "$LOG"
exec >>"$LOG/onstart.log" 2>&1
export PATH="$HOME/.local/bin:$PATH"
t0=$SECONDS
log() { echo "$(date -u +%FT%TZ) [+$((SECONDS - t0))s] $*"; }

# Restart a service if it crashes, so one bad request doesn't take the box down until the next start.
supervise() {
  local name=$1; shift
  (while true; do "$@" >>"$LOG/$name.log" 2>&1 || true; echo "$(date -u +%FT%TZ) $name exited, restarting" >>"$LOG/$name.log"; sleep 5; done) &
}

wait_http() {  # url, timeout seconds
  local deadline=$((SECONDS + $2))
  until curl -sf -o /dev/null "$1"; do
    ((SECONDS < deadline)) || { log "timeout waiting for $1"; return 1; }
    sleep 3
  done
}

# --- packages -------------------------------------------------------------
if ! command -v autossh >/dev/null; then
  log "installing apt packages"
  apt-get update -qq && apt-get install -y -qq --no-install-recommends autossh openssh-client git curl >/dev/null
fi
command -v uv >/dev/null || { log "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; }
command -v vastai >/dev/null || uv tool install -q vastai

# --- 1. vLLM ----------------------------------------------------------------
log "starting vLLM ($LLM_MODEL)"
# shellcheck disable=SC2086  # LLM_EXTRA_ARGS is a flag list on purpose
supervise vllm vllm serve "$LLM_MODEL" --served-model-name llm \
  --host 127.0.0.1 --port 8000 --api-key "$MODEL_API_KEY" \
  --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.55 \
  --enable-auto-tool-choice --tool-call-parser "$LLM_TOOL_PARSER" --reasoning-parser qwen3 \
  $LLM_EXTRA_ARGS
wait_http http://127.0.0.1:8000/health 1800   # first boot downloads the weights
log "vLLM ready"

# --- 2. speaches (STT + TTS) --------------------------------------------------
if [ ! -d /opt/speaches ]; then
  log "installing speaches $SPEACHES_REF"
  git clone -q --depth 1 --branch "$SPEACHES_REF" https://github.com/speaches-ai/speaches.git /opt/speaches
  (cd /opt/speaches && uv sync -q --frozen --no-dev \
    && uv pip install -q nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*')
fi
# The vLLM image has no system-wide cuBLAS/cuDNN (they live in vLLM's own venv); faster-whisper
# and onnxruntime-gpu need them, so point the loader at the pip-installed copies.
SITE=$(/opt/speaches/.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

log "starting speaches"
supervise speaches env -C /opt/speaches \
  UVICORN_HOST=127.0.0.1 UVICORN_PORT=8001 API_KEY="$MODEL_API_KEY" LOG_LEVEL=info ENABLE_UI=false \
  STT_MODEL_TTL=-1 TTS_MODEL_TTL=-1 WHISPER__COMPUTE_TYPE=int8_float16 \
  PRELOAD_MODELS="[\"$STT_MODEL\",\"$TTS_MODEL\"]" \
  .venv/bin/uvicorn --factory speaches.main:create_app
wait_http http://127.0.0.1:8001/health 1800   # preload = download only
log "speaches up; warming models"

# Load both models onto the GPU now: TTS a sentence, then transcribe it back.
AUTH="Authorization: Bearer $MODEL_API_KEY"
curl -sf -H "$AUTH" -H 'Content-Type: application/json' -o /tmp/warm.wav \
  -d "{\"model\":\"$TTS_MODEL\",\"voice\":\"af_heart\",\"input\":\"Warm up check, one two three.\",\"response_format\":\"wav\"}" \
  http://127.0.0.1:8001/v1/audio/speech
log "TTS warm: $(curl -sf -H "$AUTH" -F file=@/tmp/warm.wav -F model="$STT_MODEL" http://127.0.0.1:8001/v1/audio/transcriptions)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/VRAM: /' | while read -r l; do log "$l"; done

# --- 3. reverse tunnel to the VPS (SPEC §3.7) ----------------------------------
install -d -m 700 ~/.ssh
echo "$TUNNEL_KEY_B64" | base64 -d >~/.ssh/tunnel && chmod 600 ~/.ssh/tunnel
echo "$VPS_HOST_KEY" >~/.ssh/known_hosts_vps   # ssh-keyscan line; "[host]:port ..." if port != 22
log "opening tunnel to $VPS_HOST"
supervise tunnel env AUTOSSH_GATETIME=0 autossh -M 0 -N -p "$VPS_SSH_PORT" -i "$HOME/.ssh/tunnel" \
  -o UserKnownHostsFile="$HOME/.ssh/known_hosts_vps" -o StrictHostKeyChecking=yes \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:8000:127.0.0.1:8000 -R 127.0.0.1:8001:127.0.0.1:8001 "tunnel@$VPS_HOST"
log "ready"

# --- 4. idle watchdog (backstop for the VPS idle timer, SPEC §3.6) --------------
# Every user turn hits the LLM, so an unchanged request counter means nobody is talking.
# vast injects CONTAINER_ID / CONTAINER_API_KEY (an instance-scoped key) into every instance.
if [ -z "${CONTAINER_ID:-}" ] || [ -z "${CONTAINER_API_KEY:-}" ]; then
  log "CONTAINER_ID/CONTAINER_API_KEY missing: idle watchdog disabled"
  exit 0
fi
count() { curl -sf http://127.0.0.1:8000/metrics | awk '/^vllm:request_success_total/ {s += $2} END {print s + 0}'; }
last=$(count || echo 0); idle=0
while sleep 60; do
  now=$(count || echo "$last")
  if [ "$now" != "$last" ]; then last=$now; idle=0; else idle=$((idle + 1)); fi
  if ((idle >= IDLE_STOP_MINUTES)); then
    log "idle ${idle} min, stopping instance $CONTAINER_ID"
    vastai --api-key "$CONTAINER_API_KEY" stop instance "$CONTAINER_ID"
    idle=0
  fi
done
