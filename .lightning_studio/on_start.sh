#!/bin/bash
set -euo pipefail

# Lightning Studio runs this on each start. Keep it idempotent.
cd /teamspace/studios/this_studio

if [ ! -d .venv ]; then
  python -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

mkdir -p artifacts
mkdir -p .cache/huggingface

if [ ! -f .env ]; then
  cp .env.example .env
fi

echo "Lightning Studio bootstrap complete."
