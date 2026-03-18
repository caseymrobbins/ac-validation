"""
Helpers for sanity-checking episode metric inputs before large training runs.

These functions are designed for notebook use: capture one episode's raw
inputs, validate shape/content assumptions, then compute metrics with a short
report of anything suspicious.
"""

from typing import Any, Dict, List, Optional

try:
    from .metrics import compute_episode_metrics
except ImportError:
    from metrics import compute_episode_metrics


def build_episode_metric_inputs(
    utilities: List[float],
    agency_scores: List[float],
    wealth_values: List[float],
    inventory_values: List[float],
    tax_rates: List[float],
    floor_action_history: List[int],
) -> Dict[str, Any]:
    """Package one episode's raw metric inputs into a serializable dict."""
    return {
        "utilities": list(utilities),
        "agency_scores": list(agency_scores),
        "wealth_values": list(wealth_values),
        "inventory_values": list(inventory_values),
        "tax_rates": list(tax_rates),
        "floor_action_history": list(floor_action_history),
    }


def validate_episode_metric_inputs(
    episode_inputs: Dict[str, Any],
    expected_n_agents: Optional[int] = None,
) -> List[str]:
    """Return human-readable warnings for suspicious episode metric inputs."""
    warnings: List[str] = []

    vector_keys = [
        "utilities",
        "agency_scores",
        "wealth_values",
        "inventory_values",
    ]
    lengths = {
        key: len(episode_inputs.get(key, []))
        for key in vector_keys
    }

    if expected_n_agents is not None:
        for key, length in lengths.items():
            if length != expected_n_agents:
                warnings.append(
                    f"{key} has length {length}, expected {expected_n_agents}."
                )

    unique_lengths = {length for length in lengths.values()}
    if len(unique_lengths) > 1:
        warnings.append(
            f"Agent-level vectors disagree on length: {lengths}."
        )

    inventory_values = episode_inputs.get("inventory_values", [])
    wealth_values = episode_inputs.get("wealth_values", [])
    if inventory_values and wealth_values and inventory_values == wealth_values:
        warnings.append(
            "inventory_values exactly matches wealth_values; check that "
            "resource concentration is not accidentally using wealth."
        )

    utilities = episode_inputs.get("utilities", [])
    if any(value <= 0 for value in utilities):
        warnings.append(
            "One or more utilities are <= 0. This is expected to destabilize "
            "log-based planner objectives unless handled explicitly."
        )

    tax_rates = episode_inputs.get("tax_rates", [])
    if tax_rates and len(tax_rates) not in unique_lengths:
        warnings.append(
            f"tax_rates has length {len(tax_rates)}, which does not match the "
            f"agent-level vector lengths {sorted(unique_lengths)}."
        )

    if not episode_inputs.get("floor_action_history", []):
        warnings.append(
            "floor_action_history is empty; floor action diversity will be zero."
        )

    return warnings


def compute_validated_episode_metrics(
    episode_inputs: Dict[str, Any],
    expected_n_agents: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute metrics and return both the metrics and any validation warnings.

    This is intentionally lightweight so it can be called from notebook cells
    on a single saved episode trace before launching long runs.
    """
    warnings = validate_episode_metric_inputs(
        episode_inputs,
        expected_n_agents=expected_n_agents,
    )
    metrics = compute_episode_metrics(
        utilities=episode_inputs["utilities"],
        agency_scores=episode_inputs["agency_scores"],
        wealth_values=episode_inputs["wealth_values"],
        inventory_values=episode_inputs["inventory_values"],
        tax_rates=episode_inputs["tax_rates"],
        floor_action_history=episode_inputs["floor_action_history"],
    )
    return {
        "metrics": metrics,
        "warnings": warnings,
        "episode_inputs": episode_inputs,
    }
