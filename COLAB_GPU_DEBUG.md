# Colab GPU Debug Path

This repo's current Colab target is:

- Runtime: `2025.07`
- Python: `3.11.13`
- Accelerator: `GPU`

Why this pin:

- Current default Colab runtimes are Python 3.12 and break the legacy
  `gym==0.21` / AI Economist stack.
- Colab's official past-runtime FAQ currently shows `2025.07` as the available
  Python 3.11 image.
- This repo's training code now uses RLlib's newer multi-agent env API and can
  request one GPU when CUDA is available.

## 1. Start the right Colab runtime

In Colab:

1. Open `Runtime -> Change runtime type`
2. Set `Accelerator` to `GPU`
3. Set `Runtime version` to `2025.07`
4. Reconnect

## 2. Clone the repo

Run this in a fresh Colab cell:

```bash
!git clone https://github.com/caseymrobbins/ac-validation.git
%cd /content/ac-validation
```

## 3. Run the bootstrap script

```bash
!bash scripts/colab_bootstrap.sh
```

When it finishes, restart the runtime once.

## 4. Quick import sanity check

After restart:

```python
import sys, numpy, gym, ray, torch, ai_economist
print(sys.version)
print("numpy", numpy.__version__)
print("gym", gym.__version__)
print("ray", ray.__version__)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
```

Expected:

- Python 3.11.x
- `gym 0.21.0`
- `torch.cuda.is_available() == True`

## 5. Run notebook 01 first

Open [notebooks/01_setup_and_test.ipynb](C:\Users\crens\Documents\GitHub\ac-validation\notebooks\01_setup_and_test.ipynb) in Colab after the bootstrap.

Use notebook 01 for:

- environment import/setup
- stepping the env
- short smoke validation

## 6. Use a short training smoke run before a real seed

Before any 10M-step run, use a short debug training call:

```python
from training import run_training

logger = run_training(
    condition="sum",
    seed=99,
    total_timesteps=20_000,
    eval_interval=5_000,
    results_dir="../results",
)
```

Only after this completes cleanly should you launch a real seed.

## 7. Real training on Colab

In notebooks 04/05/06:

- run one seed per session
- keep `num_workers=0` for now
- let `training.py` auto-detect the GPU

If you want to force CPU or GPU manually:

- CPU: `os.environ["AC_VALIDATION_USE_GPU"] = "0"`
- GPU: `os.environ["AC_VALIDATION_USE_GPU"] = "1"`

Set that before importing `training`.

## 8. Important caveat

The planner reward swap still has to live in the AI Economist fork:

- [AI_ECONOMIST_FORK_CONTRACT.md](C:\Users\crens\Documents\GitHub\ac-validation\AI_ECONOMIST_FORK_CONTRACT.md)

`src/training.py` only logs diagnostic AC rewards; it is not the source of
truth for the planner objective.
