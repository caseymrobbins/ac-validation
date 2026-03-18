# Agency Calculus Execution Checklist

This checklist turns the March 2026 design document into an implementation
sequence that matches the current repository state.

## Goal

Produce a valid SUM vs NASH vs JAM comparison in AI Economist with:

- a real planner reward swap inside the environment/scenario reward path
- reproducible baseline behavior under SUM
- validated metric extraction
- a runtime stack that can actually be trained in the chosen environment

## Current Blocking Issues

These must be resolved before paid training runs:

1. `src/training.py` does not actually change the planner's learning reward.
   It logs a custom metric after the episode instead.
2. The repo's original notebook 01 install path does not work on current
   Colab Python 3.12 runtimes.
3. `resource_concentration` is currently computed from `wealth` during eval,
   not inventory-only values.
4. POLI inventory extraction can preserve `Wood`/`Stone` keys while the
   prerequisite computation only reads lowercase keys.
5. Several design-doc metrics are not implemented yet:
   trading balance, planner volatility, floor mobility.

## Execution Order

### Phase 0: Environment Decision

Choose one training target before doing more engineering:

- Colab modernization path:
  Port the AI Economist stack to the current Colab runtime.
- Legacy-compatible path:
  Use Kaggle or local Python 3.10/3.11 for the archived dependency stack.

Gate:
- We can create the env, import RLlib, and complete notebook 01 end-to-end in
  the target runtime.

### Phase 1: Fork + Runtime Modernization

Work in a fork of `salesforce/ai-economist`, not only in this repo.

Tasks:

- Fork AI Economist and record the fork URL/commit in this repo.
- Decide whether to:
  - modernize to current Ray/Gym-compatible packages for Colab, or
  - pin to a Python 3.10/3.11 runtime and keep the legacy stack.
- Verify the tutorial/baseline env can initialize and step cleanly.
- Document the exact runtime matrix:
  Python version, Ray version, Gym/Gymnasium version, Torch version.

Gate:
- A clean setup notebook runs without manual package surgery.

### Phase 2: Planner Reward Injection

Implement reward swapping where the planner reward is actually produced.

Tasks:

- Locate the scenario/env reward code in the AI Economist fork.
- Add planner reward modes:
  - `sum`
  - `nash`
  - `jam`
  - `jam_epsilon`
  - `jam_softmin`
- Define `u_i` once and keep it fixed across all conditions.
- Ensure JAM preserves the no-epsilon boundary in the main experiment.
- Remove any misleading comments in this repo that imply callbacks override
  planner training rewards.

Gate:
- Unit or smoke verification shows planner rewards differ by condition for the
  same env state.

### Phase 3: Baseline Reproduction

Do not start NASH/JAM until SUM is believable.

Tasks:

- Run SUM with the same scenario/config intended for the main paper.
- Verify expected baseline dynamics:
  inequality growth, concentrated outcomes, plausible planner behavior.
- Save one short debug run and one longer baseline run.

Gate:
- SUM results are directionally consistent with published AI Economist behavior.

### Phase 4: Metric Validation

Make metrics match the design doc and raw env state.

Tasks:

- Fix inventory-only concentration measurement.
- Normalize POLI inventory keys before scoring.
- Implement missing metrics:
  - trading frequency/balance
  - planner policy volatility
  - floor mobility
- Add spot-check utilities that compare logged metrics to raw episode traces.
- Decide whether `u_i` is coin, wealth, or agency proxy and document why.

Gate:
- For at least one recorded episode, each logged metric can be traced back to
  raw state/action data.

### Phase 5: Experimental Runs

Only after Phases 1-4 pass.

Tasks:

- Run 5 seeds for SUM.
- Run 5 seeds for NASH.
- Run 5 seeds for JAM.
- Evaluate every 100K steps.
- Save metrics, checkpoints, and config snapshots.

Gate:
- All runs complete with comparable configs and no silent metric failures.

### Phase 6: Limitation Experiments

Tasks:

- Run `jam_epsilon` with the documented smoothing value.
- Run `jam_softmin` with the documented temperature.
- Compare degradation against hard JAM.

Gate:
- The limitation variants produce a measurable reopening of the compensation
  window relative to hard JAM.

### Phase 7: Analysis + Reporting

Tasks:

- Plot floor utility, total utility, Gini, floor agency, concentration.
- Compute final-window statistics across seeds.
- Run significance tests at convergence.
- Write down any departures from the theory before drafting claims.

Gate:
- Every plot and claim can be reproduced from saved artifacts in `results/`.

## Immediate Repo Tasks

These are the next changes worth making in this repository:

1. Replace callback-based "reward override" messaging with code that clearly
   delegates real reward swapping to the AI Economist fork.
2. Fix POLI key normalization in `src/poli_agency.py`.
3. Fix inventory concentration input in `src/training.py`.
4. Add a small validation harness for metrics on one saved episode.
5. Add fork/runtime configuration notes to the notebooks once the fork exists.

## Suggested Working Rule

No expensive multi-seed training until all earlier gates are green.
