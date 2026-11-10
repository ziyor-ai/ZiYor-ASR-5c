#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${BASIC5_PYTHON:-python3.12}
TORCH_INDEX=${BASIC5_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
"$PYTHON" -m venv .venvs/core
.venvs/core/bin/python -m pip install -U pip
.venvs/core/bin/python -m pip install -r requirements/base.txt
.venvs/core/bin/python -m pip install --no-deps -e .
for role in qwen speaker; do
  "$PYTHON" -m venv ".venvs/$role"
  ".venvs/$role/bin/python" -m pip install -U pip
  ".venvs/$role/bin/python" -m pip install --index-url "$TORCH_INDEX" 'torch==2.10.0' 'torchaudio==2.10.0'
  ".venvs/$role/bin/python" -m pip install -r "requirements/$role.txt"
  ".venvs/$role/bin/python" -m pip install --no-deps -e .
done
if [[ ! -e config.json ]]; then cp config.example.json config.json; fi
echo 'Environments created. Next: .venvs/core/bin/python scripts/download_models.py --config config.json'
