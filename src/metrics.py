"""
Metrics for Agency Calculus empirical validation.

Logged every 100K training steps from evaluation episodes.

Metrics:
  - floor_utility:          min(u_i) across agents
  - floor_agency:           min(A_i) using POLI computation
  - gini_wealth:            Gini coefficient of wealth distribution
  - resource_concentration: max(inventory) / mean(inventory)
  - floor_tax_rate:         tax rate applied to the floor agent
  - floor_action_diversity: unique actions taken by floor agent per eval episode
  - total_utility:          sum(u_i)
  - fhi:                    Floor Harm Index = (second_lowest_agency - floor_agency) / floor_agency
"""

import numpy as np
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def gini_coefficient(values: List[float]) -> float:
    """
    Gini coefficient of a distribution.

    G = 0 means perfect equality; G = 1 means maximum inequality.
    Returns 0.0 for degenerate inputs (all zero or single agent).

    Uses the standard sorted-values formula:
      G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
    where x_i are sorted ascending and i is 1-indexed.
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr)) / (n * arr.sum()) - (n + 1) / n)


def resource_concentration(inventories: List[float]) -> float:
    """
    Resource concentration: max(inventory) / mean(inventory).

    A value of 1.0 means perfectly equal holdings.
    Higher values indicate more concentrated ownership.

    Args:
        inventories: List of total inventory values (one per agent).
    """
    arr = np.array(inventories, dtype=float)
    mean_val = arr.mean()
    if mean_val == 0:
        return 1.0
    return float(arr.max() / mean_val)


def floor_harm_index(agency_scores: List[float]) -> float:
    """
    FHI (Floor Harm Index): relative gap between floor and second-lowest agent.

    FHI = (second_lowest_agency - floor_agency) / floor_agency

    FHI = 0  -> floor agent and second-lowest are equal (floor not isolated)
    FHI > 0  -> floor agent is below the second-lowest agent
    FHI = inf -> floor agent has zero agency (divide by zero guard returns NaN)

    Args:
        agency_scores: Per-agent agency scores (from POLI computation).
    """
    if len(agency_scores) < 2:
        return 0.0
    sorted_scores = sorted(agency_scores)
    floor = sorted_scores[0]
    second = sorted_scores[1]
    if floor <= 0:
        return float("inf")
    return float((second - floor) / floor)


def action_diversity(action_history: List[int]) -> int:
    """
    Number of unique actions taken during an evaluation episode.

    Args:
        action_history: List of integer action IDs taken by the agent.
    """
    return len(set(action_history))


# ---------------------------------------------------------------------------
# Episode-level metrics aggregator
# ---------------------------------------------------------------------------

def compute_episode_metrics(
    utilities: List[float],
    agency_scores: List[float],
    wealth_values: List[float],
    inventory_values: List[float],
    tax_rates: List[float],
    floor_action_history: List[int],
) -> Dict[str, float]:
    """
    Compute all tracked metrics from a single evaluation episode.

    Args:
        utilities:            Per-agent utilities (list of n_agents floats).
        agency_scores:        Per-agent POLI agency scores.
        wealth_values:        Per-agent total wealth (coin + inventory value).
        inventory_values:     Per-agent total inventory values (for concentration).
        tax_rates:            Per-agent effective tax rates this episode.
        floor_action_history: List of actions taken by the floor (min utility) agent.

    Returns:
        Dict mapping metric name -> scalar value.
    """
    floor_idx = int(np.argmin(utilities))

    return {
        "floor_utility": float(min(utilities)),
        "floor_agency": float(min(agency_scores)),
        "gini_wealth": gini_coefficient(wealth_values),
        "resource_concentration": resource_concentration(inventory_values),
        "floor_tax_rate": float(tax_rates[floor_idx]) if tax_rates else float("nan"),
        "floor_action_diversity": action_diversity(floor_action_history),
        "total_utility": float(sum(utilities)),
        "fhi": floor_harm_index(agency_scores),
        "mean_utility": float(np.mean(utilities)),
        "mean_agency": float(np.mean(agency_scores)),
    }


# ---------------------------------------------------------------------------
# Multi-episode aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(
    episode_metrics_list: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple evaluation episodes.

    Args:
        episode_metrics_list: List of dicts from compute_episode_metrics.

    Returns:
        Dict mapping metric_name -> {"mean": ..., "std": ..., "min": ..., "max": ...}
    """
    if not episode_metrics_list:
        return {}

    keys = episode_metrics_list[0].keys()
    result = {}
    for key in keys:
        values = [ep[key] for ep in episode_metrics_list if np.isfinite(ep[key])]
        if not values:
            result[key] = {"mean": float("nan"), "std": float("nan"),
                           "min": float("nan"), "max": float("nan")}
        else:
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return result


# ---------------------------------------------------------------------------
# Training history logger
# ---------------------------------------------------------------------------

class MetricsLogger:
    """
    Accumulates metrics across training checkpoints.

    Usage:
        logger = MetricsLogger(condition="jam", seed=0)
        # ... after each eval at step N:
        logger.record(step=N, metrics=compute_episode_metrics(...))
        # At end of training:
        logger.save("results/jam_seed0_metrics.npz")
    """

    def __init__(self, condition: str, seed: int):
        self.condition = condition
        self.seed = seed
        self.records: List[Dict[str, Any]] = []

    def record(self, step: int, metrics: Dict[str, float]) -> None:
        """Record metrics at a given training step."""
        entry = {"step": step, **metrics}
        self.records.append(entry)

    def record_aggregated(self, step: int, agg: Dict[str, Dict[str, float]]) -> None:
        """Record aggregated (multi-episode) metrics at a given training step."""
        entry = {"step": step}
        for metric, stats in agg.items():
            for stat_name, val in stats.items():
                entry[f"{metric}_{stat_name}"] = val
        self.records.append(entry)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert recorded history to numpy arrays keyed by metric name."""
        if not self.records:
            return {}
        keys = self.records[0].keys()
        return {k: np.array([r[k] for r in self.records]) for k in keys}

    def save(self, path: str) -> None:
        """Save to compressed numpy archive."""
        arrays = self.to_arrays()
        arrays["condition"] = np.array([self.condition])
        arrays["seed"] = np.array([self.seed])
        np.savez_compressed(path, **arrays)
        print(f"Saved metrics to {path}")

    @classmethod
    def load(cls, path: str) -> "MetricsLogger":
        """Load from saved numpy archive."""
        data = np.load(path, allow_pickle=True)
        condition = str(data["condition"][0])
        seed = int(data["seed"][0])
        logger = cls(condition=condition, seed=seed)
        # Reconstruct records list
        steps = data["step"]
        keys = [k for k in data.files if k not in ("condition", "seed")]
        for i, step in enumerate(steps):
            record = {"step": int(step)}
            for k in keys:
                if k != "step":
                    record[k] = float(data[k][i])
            logger.records.append(record)
        return logger
