#!/bin/bash
set -euo pipefail

# Lightning Studio runs this on each start. Keep it idempotent.
cd /teamspace/studios/this_studio/dusch-maya-s2s

python -m pip install --upgrade pip
python -m pip install -e .

mkdir -p artifacts
mkdir -p /teamspace/studios/this_studio/.cache/huggingface

if [ ! -f .env ]; then
  cp .env.example .env
fi

echo "Lightning Studio bootstrap complete."
