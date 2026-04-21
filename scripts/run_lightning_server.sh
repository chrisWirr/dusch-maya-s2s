#!/bin/bash
set -euo pipefail

cd /teamspace/studios/this_studio
cd dusch-maya-s2s

export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${PORT:-${APP_PORT:-8000}}"
export OUTPUT_DIR="${OUTPUT_DIR:-/teamspace/studios/this_studio/dusch-maya-s2s/artifacts}"
export HF_HOME="${HF_HOME:-/teamspace/studios/this_studio/.cache/huggingface}"
export DEVICE="${DEVICE:-cuda}"
export TORCH_DTYPE="${TORCH_DTYPE:-float16}"

python -m maya_s2s.server
