"""
Agency Calculus Empirical Validation
Social planner reward functions for AI Economist experiments.

Three conditions:
  SUM  - utilitarian sum (baseline, permits unbounded compensation)
  NASH - Nash social welfare (permits bounded compensation)
  JAM  - floor optimization (permits zero compensation)

CRITICAL: JAM uses NO epsilon. The singularity at min_u == 0 must be
preserved. Any epsilon converts the structural guarantee into a finite cost,
destroying the theoretical property being tested.
"""

import math
import numpy as np
from typing import List


def sum_reward(utilities: List[float]) -> float:
    """
    SUM: Utilitarian social welfare (baseline).

    R = sum(u_i)

    Permits unbounded compensation: gains for any agent can offset losses
    for any other agent. This is the AI Economist default.
    """
    return float(sum(utilities))


def nash_reward(utilities: List[float], epsilon: float = 1e-4) -> float:
    """
    NASH: Nash social welfare.

    R = sum(log(u_i + epsilon))

    Epsilon is acceptable here. Nash honestly permits compensation — it
    merely bounds the exchange rate. The singularity at zero is NOT a
    structural guarantee for Nash; it is a curvature property.

    Args:
        utilities: Per-agent utility values.
        epsilon: Numerical stability offset (default 1e-4, fine for Nash).
    """
    return float(sum(math.log(u + epsilon) for u in utilities))


def jam_reward(utilities: List[float]) -> float:
    """
    JAM: Floor optimization (Agency Calculus objective).

    R = log(min(u_i))

    NO EPSILON. The singularity at min_u == 0 must be preserved.

    The gradient of log(x) as x -> 0+ is +infinity. This means the planner
    faces an infinitely costly penalty for driving any agent to zero utility,
    which is the structural guarantee that prevents floor compression.

    Adding epsilon (e.g., log(min_u + 0.1)) converts that infinite penalty
    into a finite one and re-opens the compensation window — exactly what
    the limitation experiment (08) tests.

    If training is unstable near zero: increase batch size, lower learning
    rate, or adjust worker reward scaling. Do NOT smooth this boundary.

    Args:
        utilities: Per-agent utility values.

    Returns:
        log(min(u_i)), or -1e10 if min_u <= 0.
        -1e10 acts as practical negative infinity. The planner should never
        reach this state because the gradient away from zero is infinite.
        If it does, something is wrong with the training setup.
    """
    min_u = min(utilities)
    if min_u <= 0:
        # Practical negative infinity — NOT epsilon stabilization.
        # This signals a training failure, not a design choice.
        return -1e10
    return math.log(min_u)


def jam_reward_epsilon(utilities: List[float], epsilon: float = 0.1) -> float:
    """
    JAM + epsilon (LIMITATION EXPERIMENT ONLY).

    R = log(min(u_i) + epsilon)

    This is the degraded variant used in notebook 08 to demonstrate that
    adding epsilon re-opens the compensation window and causes floor
    degradation. Do NOT use this for the main experiment.

    Args:
        utilities: Per-agent utility values.
        epsilon: Smoothing offset. Default 0.1 as specified in Paper C.
    """
    min_u = min(utilities)
    return math.log(min_u + epsilon)


def softmin(x: List[float], tau: float) -> float:
    """
    Soft minimum with temperature parameter tau.

    Used in the softmin limitation experiment (notebook 08) to show that
    approximating the minimum reopens a compensation window of size ~tau.

    Args:
        x: Values to take the soft minimum of.
        tau: Temperature. Lower tau -> harder minimum. tau -> 0 recovers min().
    """
    arr = np.array(x, dtype=float)
    shifted = -arr / tau
    shifted -= shifted.max()  # numerical stability
    w = np.exp(shifted)
    w /= w.sum()
    return float(np.sum(w * arr))


def jam_reward_softmin(utilities: List[float], tau: float = 0.5) -> float:
    """
    JAM + softmin (LIMITATION EXPERIMENT ONLY).

    R = log(softmin(u_i, tau))

    Demonstrates that approximating the floor reopens a compensation window
    proportional to tau. Do NOT use for the main experiment.
    """
    sm = softmin(utilities, tau)
    if sm <= 0:
        return -1e10
    return math.log(sm)


# Reward function registry for clean dispatch in training notebooks
REWARD_FUNCTIONS = {
    "sum": sum_reward,
    "nash": nash_reward,
    "jam": jam_reward,
    # Limitation variants
    "jam_epsilon": jam_reward_epsilon,
    "jam_softmin": jam_reward_softmin,
}


def get_reward_fn(name: str):
    """Return reward function by name. Raises KeyError for unknown names."""
    if name not in REWARD_FUNCTIONS:
        raise KeyError(
            f"Unknown reward function '{name}'. "
            f"Valid options: {list(REWARD_FUNCTIONS.keys())}"
        )
    return REWARD_FUNCTIONS[name]
