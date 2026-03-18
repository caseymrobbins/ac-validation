# AI Economist Fork Contract

This repository is the experiment layer. The actual planner reward swap must be
implemented in the AI Economist fork.

## Required Fork Behavior

The fork should expose a condition-controlled planner objective in the real
environment/scenario reward path, not only in notebook-side wrappers or RLlib
callbacks.

Recommended config surface:

```python
env_config = {
    "planner_objective": "sum",   # or "nash", "jam", "jam_epsilon", "jam_softmin"
    "planner_utility_source": "coin",
}
```

## Required Conditions

- `sum`: `sum(u_i)`
- `nash`: `sum(log(u_i + epsilon))`
- `jam`: `log(min(u_i))`
- `jam_epsilon`: `log(min(u_i) + epsilon)`
- `jam_softmin`: `log(softmin(u_i, tau))`

## Non-Negotiable Rules

- The main JAM condition must use no epsilon.
- The utility definition `u_i` must be fixed across all conditions.
- The planner reward must be computed in the same place the planner policy
  receives its learning signal.
- Limitation variants should be opt-in and clearly separated from the main JAM
  condition.

## Suggested Verification

Before running long experiments, verify all of the following in the fork:

1. For one fixed env state, planner rewards differ across `sum`, `nash`, and `jam`.
2. A short training smoke test produces distinct planner reward traces by condition.
3. The fork logs the active `planner_objective` into results/checkpoints.
4. Notebook 04/05/06 config matches the objective actually used in the fork.

## Repo Boundary

What this repo does:

- reward-function definitions
- metric extraction and validation
- plotting and analysis
- notebook orchestration

What the fork must do:

- actual planner reward replacement
- runtime compatibility with the chosen training environment
- scenario/env integration for condition switching
