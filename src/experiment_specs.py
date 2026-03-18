"""
Shared experiment-condition metadata for Agency Calculus runs.

This keeps the condition registry in one place so notebooks, validation, and
the eventual AI Economist fork can reference the same names and semantics.
"""

from typing import Dict, List


CONDITION_SPECS: Dict[str, Dict[str, object]] = {
    "sum": {
        "label": "SUM",
        "family": "main",
        "description": "Utilitarian planner objective: sum(u_i).",
        "requires_fork_reward_swap": True,
    },
    "nash": {
        "label": "NASH",
        "family": "main",
        "description": "Nash social welfare: sum(log(u_i + epsilon)).",
        "requires_fork_reward_swap": True,
    },
    "jam": {
        "label": "JAM",
        "family": "main",
        "description": "Hard floor objective: log(min(u_i)) with no epsilon.",
        "requires_fork_reward_swap": True,
    },
    "jam_epsilon": {
        "label": "JAM + epsilon",
        "family": "limitation",
        "description": "Limitation variant: log(min(u_i) + epsilon).",
        "requires_fork_reward_swap": True,
    },
    "jam_softmin": {
        "label": "JAM + softmin",
        "family": "limitation",
        "description": "Limitation variant: log(softmin(u_i, tau)).",
        "requires_fork_reward_swap": True,
    },
}


def list_conditions(family: str = "") -> List[str]:
    """Return known condition names, optionally filtered by family."""
    if not family:
        return list(CONDITION_SPECS.keys())
    return [
        name for name, spec in CONDITION_SPECS.items()
        if spec.get("family") == family
    ]


def validate_condition_name(condition: str) -> str:
    """Raise a clear error for unknown experiment conditions."""
    if condition not in CONDITION_SPECS:
        valid = ", ".join(list_conditions())
        raise ValueError(f"Unknown condition '{condition}'. Valid options: {valid}")
    return condition
