"""
POLI Agency Proxy Computation for AI Economist

Maps AI Economist simulation state variables to the five POLI dimensions
defined in Paper B (Agency Calculus), then combines them into per-agent
agency scores.

POLI Dimensions:
  P - Prerequisites: access to foundational resources
  O - Options:       breadth of available choices
  L - Levers:        number of actions available this step
  I - Impact:        realized magnitude of state changes
  K - Knowledge:     fraction of world state observable

Agency_i = geometric_mean(P_i, O_i, L_i, I_i) * K_i

All dimensions are normalized to [0, 1] before combination.
"""

import numpy as np
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _safe_normalize(value: float, max_value: float) -> float:
    """Clip to [0,1] after dividing by max_value. Avoids division by zero."""
    if max_value <= 0:
        return 0.0
    return float(np.clip(value / max_value, 0.0, 1.0))


def geometric_mean(values: List[float]) -> float:
    """
    Geometric mean of a list of values.

    Returns 0.0 if any value is <= 0, which ensures the agency score is
    zero when any capacity dimension is absent.
    """
    arr = np.array(values, dtype=float)
    if np.any(arr <= 0):
        return 0.0
    return float(np.prod(arr) ** (1.0 / len(arr)))


def normalize_inventory(inventory: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Canonicalize inventory keys to lowercase resource names.

    AI Economist state sometimes exposes title-cased keys (`Wood`, `Stone`)
    while the POLI helpers expect lowercase names. Normalizing once here keeps
    downstream metric computation consistent.
    """
    normalized = {"wood": 0.0, "stone": 0.0}
    if not isinstance(inventory, dict):
        return normalized

    for key, value in inventory.items():
        key_lower = str(key).lower()
        if key_lower in normalized:
            normalized[key_lower] = float(value)
    return normalized


# ---------------------------------------------------------------------------
# POLI dimension extractors
# ---------------------------------------------------------------------------

def compute_prerequisites(
    coin: float,
    inventory: Dict[str, float],
    max_coin: float = 500.0,
    max_inventory_value: float = 200.0,
    stone_price: float = 1.0,
    wood_price: float = 1.0,
) -> float:
    """
    P - Prerequisites: economic foundation for participation.

    Proxy: normalized coin + normalized inventory value.
    We take their average so P lives in [0, 1].

    Args:
        coin: Agent's current coin holdings.
        inventory: Dict of resource_name -> quantity.
        max_coin: Normalization ceiling for coin.
        max_inventory_value: Normalization ceiling for inventory value.
        stone_price, wood_price: Prices for inventory valuation.
    """
    inventory = normalize_inventory(inventory)
    inv_value = (
        inventory.get("stone", 0) * stone_price
        + inventory.get("wood", 0) * wood_price
    )
    p_coin = _safe_normalize(coin, max_coin)
    p_inv = _safe_normalize(inv_value, max_inventory_value)
    return (p_coin + p_inv) / 2.0


def compute_options(
    reachable_tiles: int,
    tradeable_goods: int,
    max_tiles: int = 100,
    max_goods: int = 4,
) -> float:
    """
    O - Options: breadth of feasible choices.

    Proxy: normalized reachable tiles + normalized number of tradeable goods.

    Args:
        reachable_tiles: Number of map tiles the agent can currently reach.
        tradeable_goods: Number of resource types available for trade.
        max_tiles: Normalization ceiling.
        max_goods: Normalization ceiling.
    """
    o_tiles = _safe_normalize(reachable_tiles, max_tiles)
    o_goods = _safe_normalize(tradeable_goods, max_goods)
    return (o_tiles + o_goods) / 2.0


def compute_levers(
    available_actions: int,
    max_actions: int = 50,
) -> float:
    """
    L - Levers: number of actions available this step.

    Proxy: normalized action space size available to the agent at this step.

    Args:
        available_actions: Number of valid actions the agent can take now.
        max_actions: Normalization ceiling.
    """
    return _safe_normalize(available_actions, max_actions)


def compute_impact(
    income_change: float,
    max_income_change: float = 100.0,
) -> float:
    """
    I - Impact: realized magnitude of state changes.

    Proxy: |income change from last period|, normalized.
    Using absolute value captures impact in either direction.

    Args:
        income_change: Change in income since last period.
        max_income_change: Normalization ceiling.
    """
    return _safe_normalize(abs(income_change), max_income_change)


def compute_knowledge(
    observable_fraction: float,
) -> float:
    """
    K - Knowledge: fraction of world state observable to agent.

    Already in [0, 1]; just clipped for safety.

    Args:
        observable_fraction: Fraction of map tiles visible to agent.
    """
    return float(np.clip(observable_fraction, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Full agency score
# ---------------------------------------------------------------------------

def compute_agency(
    coin: float,
    inventory: Dict[str, float],
    reachable_tiles: int,
    tradeable_goods: int,
    available_actions: int,
    income_change: float,
    observable_fraction: float,
    # Normalization ceilings (can be tuned per run)
    max_coin: float = 500.0,
    max_inventory_value: float = 200.0,
    max_tiles: int = 100,
    max_goods: int = 4,
    max_actions: int = 50,
    max_income_change: float = 100.0,
) -> Dict[str, float]:
    """
    Compute POLI agency score for a single agent.

    Agency_i = geometric_mean(P_i, O_i, L_i, I_i) * K_i

    Returns a dict with all intermediate scores for logging/debugging.
    """
    P = compute_prerequisites(coin, inventory, max_coin, max_inventory_value)
    O = compute_options(reachable_tiles, tradeable_goods, max_tiles, max_goods)
    L = compute_levers(available_actions, max_actions)
    I = compute_impact(income_change, max_income_change)
    K = compute_knowledge(observable_fraction)

    capacity = geometric_mean([P, O, L, I])
    agency = capacity * K

    return {
        "prerequisites": P,
        "options": O,
        "levers": L,
        "impact": I,
        "knowledge": K,
        "capacity": capacity,
        "agency": agency,
    }


# ---------------------------------------------------------------------------
# AI Economist state extraction
# ---------------------------------------------------------------------------

def extract_agent_state_from_obs(
    agent_idx: int,
    obs: Dict[str, Any],
    env_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract raw state variables for a single agent from AI Economist
    observation dict. This handles the AI Economist's observation structure.

    The AI Economist observation for each worker agent contains:
      - 'flat': flattened observation vector
      - Various sub-dicts depending on env configuration

    This function provides a best-effort extraction with sensible defaults
    for dimensions not directly observable. Callers should adapt this to
    the specific environment configuration being used.

    Args:
        agent_idx: Integer index of the agent (0-3 for 4 worker agents).
        obs: Observation dict from the environment step.
        env_state: Optional raw environment state for additional variables.

    Returns:
        Dict with keys: coin, inventory, reachable_tiles, tradeable_goods,
        available_actions, income_change, observable_fraction.
    """
    agent_key = str(agent_idx)

    # Default values — override with actual observations where available
    state = {
        "coin": 0.0,
        "inventory": {"wood": 0, "stone": 0},
        "reachable_tiles": 25,      # conservative default (5x5 neighborhood)
        "tradeable_goods": 2,        # wood and stone
        "available_actions": 10,     # default action space size
        "income_change": 0.0,
        "observable_fraction": 0.25, # 1/n_agents default
    }

    if obs is None:
        return state

    # AI Economist stores per-agent obs under string keys
    agent_obs = obs.get(agent_key, obs.get(agent_idx, {}))

    if isinstance(agent_obs, dict):
        # Coin / endowment
        if "coin" in agent_obs:
            state["coin"] = float(agent_obs["coin"])
        elif "endow" in agent_obs:
            # Endowment array: [coin, wood, stone, ...]
            endow = agent_obs["endow"]
            if hasattr(endow, "__len__") and len(endow) >= 1:
                state["coin"] = float(endow[0])
                if len(endow) >= 3:
                    state["inventory"] = normalize_inventory({
                        "wood": float(endow[1]),
                        "stone": float(endow[2]),
                    })

        # Inventory
        if "inventory" in agent_obs:
            inv = agent_obs["inventory"]
            state["inventory"] = normalize_inventory(inv)

    # Use env_state for richer extraction when available
    if env_state is not None:
        agent_state = env_state.get("agents", {}).get(agent_key, {})
        if "coin" in agent_state:
            state["coin"] = float(agent_state["coin"])
        if "inventory" in agent_state:
            state["inventory"] = normalize_inventory(agent_state["inventory"])
        if "income_change" in agent_state:
            state["income_change"] = float(agent_state["income_change"])

    return state


def compute_agency_from_obs(
    agent_idx: int,
    obs: Dict[str, Any],
    env_state: Optional[Dict[str, Any]] = None,
    norm_params: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Full pipeline: extract state from AI Economist obs, compute POLI agency.

    Args:
        agent_idx: Agent index (0-3).
        obs: Environment observation dict.
        env_state: Optional raw environment state for richer extraction.
        norm_params: Optional dict of normalization ceilings to override
                     defaults (keys: max_coin, max_inventory_value, etc.)

    Returns:
        Agency dict with keys: prerequisites, options, levers, impact,
        knowledge, capacity, agency.
    """
    raw = extract_agent_state_from_obs(agent_idx, obs, env_state)
    params = norm_params or {}
    return compute_agency(
        coin=raw["coin"],
        inventory=raw["inventory"],
        reachable_tiles=raw["reachable_tiles"],
        tradeable_goods=raw["tradeable_goods"],
        available_actions=raw["available_actions"],
        income_change=raw["income_change"],
        observable_fraction=raw["observable_fraction"],
        **params,
    )


def compute_all_agency_scores(
    n_agents: int,
    obs: Dict[str, Any],
    env_state: Optional[Dict[str, Any]] = None,
    norm_params: Optional[Dict[str, float]] = None,
) -> List[Dict[str, float]]:
    """
    Compute POLI agency scores for all worker agents.

    Args:
        n_agents: Number of worker agents (typically 4).
        obs: Environment observation dict.
        env_state: Optional raw environment state.
        norm_params: Optional normalization ceiling overrides.

    Returns:
        List of agency dicts, one per agent (indexed 0 to n_agents-1).
    """
    return [
        compute_agency_from_obs(i, obs, env_state, norm_params)
        for i in range(n_agents)
    ]
