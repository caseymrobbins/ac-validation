# Agency Calculus Empirical Validation

**Paper C: AI Economist Experiment**

Empirical validation of Agency Calculus (AC) predictions using the
[Salesforce AI Economist](https://github.com/salesforce/ai-economist) multi-agent
economic simulation. This is the third paper in a three-paper series on AI alignment.

## Context

**Paper A** ("The Structural Failure of Aggregate Optimization") proves that
sum-based objectives structurally produce every known alignment failure through
a compensation mechanism.

**Paper B** ("Agency Calculus: A Floor Optimization Architecture") presents
the solution: the JAM objective `R = log(min(u_i))`, whose infinite gradient at
zero makes floor compression infinitely costly.

**Paper C (this)** validates those structural predictions empirically.

## The Experiment

Three social planner objectives are tested in the AI Economist simulation:

| Condition | Objective | Compensation |
|-----------|-----------|--------------|
| SUM | `R = Σu_i` | Unbounded |
| NASH | `R = Σ log(u_i + ε)` | Bounded |
| JAM | `R = log(min(u_i))` | Zero (NO epsilon) |

5 seeds × 10M timesteps × 3 conditions = 15 training runs.

Worker agent rewards are identical across conditions. Only the planner objective changes.

## Predicted Results

| Metric | SUM | NASH | JAM |
|--------|-----|------|-----|
| Floor utility | Declines | Stable | **Rises** |
| Gini coefficient | High | Moderate | Low |
| Resource concentration | High | Moderate | Low |
| Total system utility | **Highest** | High | Lower but stable |
| Tax on floor agent | Extractive | Moderate | Minimal |

The critical result: **JAM produces lower total utility but higher floor utility.**
This is the compensation mechanism. Under SUM the planner sacrifices the floor
for aggregate gain. Under JAM it cannot.

## Repository Structure

```
ac-validation/
  README.md
  requirements.txt
  notebooks/
    01_setup_and_test.ipynb       # Fork, install, verify baseline works
    02_implement_rewards.ipynb    # SUM/NASH/JAM reward functions + analysis
    03_implement_poli.ipynb       # POLI agency proxy computation
    04_training_sum.ipynb         # SUM baseline (5 seeds)
    05_training_nash.ipynb        # NASH condition (5 seeds)
    06_training_jam.ipynb         # JAM condition (5 seeds)
    07_analysis.ipynb             # All plots, statistics, comparisons
    08_limitations.ipynb          # Epsilon and softmin degradation experiments
  src/
    ac_rewards.py                 # Three reward functions
    poli_agency.py                # POLI score computation from sim state
    metrics.py                    # Gini, FHI, resource concentration, etc.
    plotting.py                   # Standard comparison plots
    training.py                   # Shared training loop (RLlib + AI Economist)
  results/                        # Saved metrics, checkpoints, figures
```

## Runtime Compatibility

This project currently depends on an older AI Economist stack that expects
Python 3.10/3.11-era packages, especially `gym==0.21` and `ray==2.3.0`.
Current Google Colab runtimes use Python 3.12, so notebook 01 will not install
cleanly there.

Execution checklist: see `EXECUTION_CHECKLIST.md` for the recommended order of
operations and gating criteria before running paid training jobs.
Fork contract: see `AI_ECONOMIST_FORK_CONTRACT.md` for what must be implemented
inside the AI Economist fork before SUM/NASH/JAM training runs are valid.

Use one of these instead:

- Kaggle notebook runtime with Python 3.10/3.11
- Local environment with Python 3.10 or 3.11

## Quick Start

```python
# 1. Create a Python 3.10/3.11 environment, then install
!pip install git+https://github.com/salesforce/ai-economist.git
!pip install 'ray[rllib]==2.3.0' torch

# 2. Clone repo
!git clone https://github.com/caseymrobbins/ac-validation.git
import sys; sys.path.insert(0, 'ac-validation/src')

# 3. Run notebook 01 to verify setup
# 4. Run notebooks 04-06 for training (one per session)
# 5. Run notebook 07 for analysis
```

## Critical Implementation Notes

### JAM Reward — NO EPSILON

```python
def jam_reward(utilities):
    min_u = min(utilities)
    if min_u <= 0:
        return -1e10  # Practical negative infinity, NOT epsilon
    return math.log(min_u)
```

**Do NOT add epsilon to JAM.** `log(min(u) + epsilon)` converts the infinite
gradient into a finite one (1/epsilon), destroying the structural guarantee.
If training is unstable, increase batch size or lower learning rate.

The limitation experiment (notebook 08) explicitly tests what epsilon does —
that is the point of adding it to the limitation variant.

### POLI Agency Proxy

```
Agency_i = geometric_mean(P, O, L, I) × K

P = Prerequisites: coin + inventory value (normalized)
O = Options:       reachable tiles + tradeable goods
L = Levers:        available actions this step
I = Impact:        |income change| (normalized)
K = Knowledge:     observable fraction of map
```

### Metrics Logged Every 100K Steps

- `floor_utility`: min(u_i) — **the primary outcome variable**
- `floor_agency`: min(POLI agency) across agents
- `gini_wealth`: Gini coefficient of wealth
- `resource_concentration`: max(inventory) / mean(inventory)
- `floor_tax_rate`: tax rate on floor agent
- `floor_action_diversity`: unique actions by floor agent per episode
- `total_utility`: Σu_i
- `fhi`: Floor Harm Index = (second_lowest - floor) / floor

## Related Work

- AI Economist: Zheng et al., "The AI Economist: Taxation policy design via
  two-level deep multiagent reinforcement learning," *Science Advances*, 2022
- Demo repository (greedy-optimizer version): https://github.com/caseymrobbins/Ac_demo
- Paper A: "The Structural Failure of Aggregate Optimization" — K.C.M. Robbins
- Paper B: "Agency Calculus: A Floor Optimization Architecture for AI Alignment" — K.C.M. Robbins
