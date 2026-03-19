#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import sys

major, minor = sys.version_info[:2]
if (major, minor) != (3, 11):
    raise SystemExit(
        "This Colab bootstrap targets Python 3.11.\n"
        "In Colab, choose Runtime -> Change runtime type -> Runtime version -> 2025.07,\n"
        "then reconnect and rerun this script."
    )

print(f"Python runtime OK: {sys.version}")
PY

python -m pip install --upgrade "pip<24.1" "setuptools<66" "wheel<0.40"
python -m pip install "numpy<2.0"
python -m pip install --no-build-isolation "gym==0.21.0"

# Keep the Colab GPU torch that ships with the pinned runtime.
python -m pip install "ray[rllib]==2.48.0"
python -m pip install pycryptodome GPUtil matplotlib seaborn pandas scipy tqdm ipywidgets cloudpickle psutil
python -m pip install --no-deps git+https://github.com/caseymrobbins/ai-economist.git@codex/ac-planner-objectives

python - <<'PY'
import sys
import numpy as np
import gym
import ray
import torch
import ai_economist

print("Bootstrap complete.")
print(f"Python: {sys.version}")
print(f"numpy: {np.__version__}")
print(f"gym: {gym.__version__}")
print(f"ray: {ray.__version__}")
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"ai_economist import: OK")
PY

touch .colab_bootstrap_complete

echo
echo "Restart the Colab runtime now, then run the notebook cells from the top."
