"""
Training loop for Agency Calculus AI Economist experiments.

Shared by notebooks 04 (SUM), 05 (NASH), 06 (JAM), and 08 (limitations).

Architecture:
  - 4 worker agents + 1 planner (AI Economist default)
  - PPO via RLlib
  - Worker rewards: individual utility (unchanged across conditions)
  - Planner reward: must be overridden inside the AI Economist fork/env
  - Evaluation: every EVAL_INTERVAL steps, run EVAL_EPISODES episodes
  - Logging: MetricsLogger saves to results/
"""

import os
import sys
import math
import time
import numpy as np
from typing import Optional, Dict, Any, List

# Allow import from src/ when running from notebooks/
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from .ac_rewards import get_reward_fn
    from .experiment_specs import validate_condition_name
    from .poli_agency import compute_all_agency_scores
    from .metrics import compute_episode_metrics, aggregate_metrics, MetricsLogger
except ImportError:
    from ac_rewards import get_reward_fn
    from experiment_specs import validate_condition_name
    from poli_agency import compute_all_agency_scores
    from metrics import compute_episode_metrics, aggregate_metrics, MetricsLogger

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS = 10_000_000   # 10M per run (reduce for debugging)
EVAL_INTERVAL = 100_000        # evaluate every 100K steps
EVAL_EPISODES = 100            # eval episodes per checkpoint
N_AGENTS = 4
N_SEEDS = 5
TRAIN_BATCH_SIZE = 4000
ROLLOUT_FRAGMENT_LENGTH = 200

# Environment configuration (shared across all conditions)
BASE_ENV_CONFIG = {
    'scenario_name': 'layout_from_file/simple_wood_and_stone',
    'components': [
        {'Build': {'skill_dist': 'pareto', 'payment_max_skill_multiplier': 3}},
        {'ContinuousDoubleAuction': {'max_num_orders': 5}},
        {'Gather': {}},
    ],
    'env_layout_file': 'quadrant_25x25_20each_30clump.txt',
    'starting_agent_coin': 10,
    'n_agents': N_AGENTS,
    'world_size': [25, 25],
    'episode_length': 1000,
    'multi_action_mode_agents': False,
    'multi_action_mode_planner': True,
    'flatten_observations': False,
    'flatten_masks': True,
    'allow_observation_scaling': True,
    'mixing_weight_gini_vs_coin': 0.0,
}


# ---------------------------------------------------------------------------
# Environment + reward analysis helpers
# ---------------------------------------------------------------------------

def make_env(seed: int = 0, env_config_overrides: Optional[Dict[str, Any]] = None):
    """Create a fresh AI Economist environment instance."""
    from ai_economist import foundation
    env_config = dict(BASE_ENV_CONFIG)
    if env_config_overrides:
        env_config.update(env_config_overrides)
    env = foundation.make_env_instance(**env_config)
    # AI Economist requires strictly positive integer seeds.
    normalized_seed = int(seed)
    if normalized_seed <= 0:
        normalized_seed = abs(normalized_seed) + 1
    env.seed(normalized_seed)
    return env


def _agent_coin_endowment(agent) -> float:
    """Safely read an agent's current Coin endowment from AI Economist state."""
    try:
        return float(agent.total_endowment("Coin"))
    except Exception:
        inventory = getattr(agent, "inventory", {}) or {}
        escrow = getattr(agent, "escrow", {}) or {}
        return float(inventory.get("Coin", 0.0)) + float(escrow.get("Coin", 0.0))


def _agent_inventory_amount(agent, resource: str) -> float:
    """Safely read an agent inventory resource amount."""
    try:
        return float(agent.inventory.get(resource, 0.0))
    except Exception:
        inventory = getattr(agent, "inventory", {}) or {}
        return float(inventory.get(resource, 0.0))


def compute_planner_reward_from_env(env, condition: str) -> float:
    """
    Compute the planner reward implied by the current environment state.

    Utility proxy: per-agent coin holdings (end-of-episode metric).
    We ensure utilities are > 0 for log-based objectives.

    NOTE:
        This helper is for analysis and sanity checks only. It does not change
        the planner's training reward by itself. The real reward swap must
        happen in the AI Economist scenario/env code path.
    """
    reward_fn = get_reward_fn(condition)
    utilities = []
    for i in range(N_AGENTS):
        agent = env.get_agent(str(i))
        coin = _agent_coin_endowment(agent)
        utilities.append(max(coin, 1e-8))
    return reward_fn(utilities)


def extract_utilities(env) -> List[float]:
    """Extract per-agent coin utility from environment state."""
    utils = []
    for i in range(N_AGENTS):
        agent = env.get_agent(str(i))
        coin = _agent_coin_endowment(agent)
        utils.append(max(coin, 1e-8))
    return utils


def extract_wealth(env) -> List[float]:
    """Extract per-agent total wealth (coin + inventory value)."""
    wealth = []
    for i in range(N_AGENTS):
        agent = env.get_agent(str(i))
        coin = _agent_coin_endowment(agent)
        wood = _agent_inventory_amount(agent, 'Wood')
        stone = _agent_inventory_amount(agent, 'Stone')
        wealth.append(coin + wood + stone)
    return wealth


def extract_inventory_values(env) -> List[float]:
    """Extract per-agent inventory-only totals for concentration metrics."""
    inventory_values = []
    for i in range(N_AGENTS):
        agent = env.get_agent(str(i))
        wood = _agent_inventory_amount(agent, 'Wood')
        stone = _agent_inventory_amount(agent, 'Stone')
        inventory_values.append(wood + stone)
    return inventory_values


def extract_tax_rates(env) -> List[float]:
    """Extract effective tax rates per agent from the planner."""
    try:
        # AI Economist stores tax brackets in the PeriodicBracketTax component
        tax_comp = env.get_component('PeriodicBracketTax')
        rates = []
        for i in range(N_AGENTS):
            rate = float(tax_comp.marginal_rate(env.get_agent(str(i))))
            rates.append(rate)
        return rates
    except Exception:
        return [0.0] * N_AGENTS


def detect_training_device() -> Dict[str, Any]:
    """
    Detect whether PPO should request a GPU.

    Behavior:
      - Default: use 1 GPU when torch reports CUDA is available.
      - Override with AC_VALIDATION_USE_GPU=0/1.
    """
    override = os.environ.get('AC_VALIDATION_USE_GPU')

    try:
        import torch
    except Exception as exc:
        return {
            'num_gpus': 0,
            'device': 'cpu',
            'reason': f'torch import failed: {exc}',
        }

    if override is not None:
        use_gpu = override.strip().lower() not in {'0', 'false', 'no', ''}
        reason = f'AC_VALIDATION_USE_GPU={override}'
    else:
        use_gpu = bool(torch.cuda.is_available())
        reason = 'torch.cuda.is_available()'

    if use_gpu:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = 'CUDA GPU'
        return {
            'num_gpus': 1,
            'device': gpu_name,
            'reason': reason,
        }

    return {
        'num_gpus': 0,
        'device': 'cpu',
        'reason': reason,
    }


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------

def run_eval_episodes(env, policy_dict: dict, condition: str,
                      n_episodes: int = EVAL_EPISODES) -> Dict[str, float]:
    """
    Run evaluation episodes and compute aggregated metrics.

    Args:
        env: AI Economist environment.
        policy_dict: RLlib policy dictionary for action sampling.
        condition: Reward condition name (for planner reward computation).
        n_episodes: Number of evaluation episodes.

    Returns:
        Aggregated metric dict (mean/std/min/max per metric).
    """
    episode_results = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = {'__all__': False}
        floor_actions = []
        step_count = 0

        while not done.get('__all__', False) and step_count < 1000:
            # Sample actions from policies
            actions = {}
            for i in range(N_AGENTS):
                agent_id = str(i)
                if 'worker' in policy_dict:
                    agent_obs = obs.get(agent_id, {})
                    if isinstance(agent_obs, dict) and 'flat' in agent_obs:
                        agent_obs = agent_obs['flat']
                    action, _, _ = policy_dict['worker'].compute_single_action(
                        agent_obs
                    )
                    action = int(np.asarray(action).item())
                else:
                    agent = env.get_agent(agent_id)
                    action = int(np.random.randint(0, agent.action_spaces))
                actions[agent_id] = action

            # Planner action
            if 'planner' in policy_dict:
                planner_obs = obs.get('p', {})
                if isinstance(planner_obs, dict) and 'flat' in planner_obs:
                    planner_obs = planner_obs['flat']
                p_action, _, _ = policy_dict['planner'].compute_single_action(
                    planner_obs
                )
                if isinstance(p_action, np.ndarray):
                    p_action = p_action.tolist()
                p_action = [int(x) for x in p_action]
            else:
                planner = env.get_agent('p')
                p_action = [int(np.random.randint(0, n)) for n in planner.action_spaces]
            actions['p'] = p_action

            obs, rewards, done, info = env.step(actions)
            step_count += 1

            # Track floor agent actions
            utilities = extract_utilities(env)
            floor_idx = int(np.argmin(utilities))
            if isinstance(actions.get(str(floor_idx)), dict):
                # Multi-action-mode: take first action value
                action_vals = list(actions[str(floor_idx)].values())
                if action_vals:
                    floor_actions.append(int(action_vals[0]))
            else:
                floor_actions.append(int(actions.get(str(floor_idx), 0)))

        # End of episode: compute metrics
        utilities = extract_utilities(env)
        wealth = extract_wealth(env)
        inventory_values = extract_inventory_values(env)
        tax_rates = extract_tax_rates(env)
        agency_scores_raw = compute_all_agency_scores(N_AGENTS, obs)
        agency_scores = [a['agency'] for a in agency_scores_raw]

        ep_metrics = compute_episode_metrics(
            utilities=utilities,
            agency_scores=agency_scores,
            wealth_values=wealth,
            inventory_values=inventory_values,
            tax_rates=tax_rates,
            floor_action_history=floor_actions,
        )
        episode_results.append(ep_metrics)

    return aggregate_metrics(episode_results)


# ---------------------------------------------------------------------------
# RLlib training run
# ---------------------------------------------------------------------------

def run_training(
    condition: str,
    seed: int,
    total_timesteps: int = TOTAL_TIMESTEPS,
    eval_interval: int = EVAL_INTERVAL,
    results_dir: str = '../results',
    verbose: bool = True,
) -> MetricsLogger:
    """
    Full training run for one condition + seed.

    Args:
        condition: Experiment condition name (for example 'sum', 'nash',
            'jam', 'jam_epsilon', or 'jam_softmin')
        seed: Random seed (0-4)
        total_timesteps: Total training steps
        eval_interval: Steps between evaluations
        results_dir: Where to save checkpoints and metrics
        verbose: Print progress

    Returns:
        MetricsLogger with full training history.
    """
    condition = validate_condition_name(condition)

    import ray
    from ray.rllib.algorithms.ppo import PPO

    os.makedirs(results_dir, exist_ok=True)

    if verbose:
        print(f'\n{"="*60}')
        print(f'Training: condition={condition}, seed={seed}')
        print(f'Total steps: {total_timesteps:,}  Eval every: {eval_interval:,}')
        print(f'{"="*60}')

    device_info = detect_training_device()
    ray_num_cpus = max(2, min(8, os.cpu_count() or 2))

    if verbose:
        print(
            f'Execution device: {device_info["device"]} '
            f'(num_gpus={device_info["num_gpus"]}, source={device_info["reason"]})'
        )

    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=ray_num_cpus,
            num_gpus=device_info['num_gpus'],
            log_to_driver=False,
        )

    logger = MetricsLogger(condition=condition, seed=seed)

    # Build trainer config
    effective_env_config = dict(
        BASE_ENV_CONFIG,
        planner_reward_type=condition,
        planner_utility_source='coin_endowment',
        # RLlib's default torch vision models do not handle the raw [2, 25, 25]
        # map tensor shape cleanly here. Use flattened observations for PPO.
        flatten_observations=True,
    )

    try:
        from gymnasium.spaces import Box, Dict as SpaceDict, Discrete, MultiDiscrete
    except ImportError:
        from gym.spaces import Box, Dict as SpaceDict, Discrete, MultiDiscrete
    from ray.rllib.env.multi_agent_env import MultiAgentEnv

    sample_env = make_env(seed=seed, env_config_overrides=effective_env_config)
    sample_obs = sample_env.reset()
    worker_flat_obs = np.asarray(sample_obs['0']['flat'], dtype=np.float32)
    planner_flat_obs = np.asarray(sample_obs['p']['flat'], dtype=np.float32)
    worker_obs_space = Box(
        low=-1e20, high=1e20, shape=worker_flat_obs.shape, dtype=np.float32
    )
    planner_obs_space = Box(
        low=-1e20, high=1e20, shape=planner_flat_obs.shape, dtype=np.float32
    )
    worker_action_space = Discrete(int(sample_env.get_agent('0').action_spaces))
    planner_action_space = MultiDiscrete(
        np.array(sample_env.get_agent('p').action_spaces, dtype=np.int64)
    )

    def _policy_obs(agent_obs):
        if isinstance(agent_obs, dict) and 'flat' in agent_obs:
            return np.asarray(agent_obs['flat'], dtype=np.float32)
        return np.asarray(agent_obs, dtype=np.float32)

    class FoundationMultiAgentEnv(MultiAgentEnv):
        """Minimal RLlib-compatible wrapper for AI Economist PPO training."""

        def __init__(self, config=None):
            super().__init__()
            config = dict(config or {})
            seed_override = int(config.pop('seed', seed))
            self.env = make_env(seed=seed_override, env_config_overrides=config)
            self._agent_ids = {str(i) for i in range(self.env.n_agents)} | {'p'}
            self.observation_spaces = {
                **{str(i): worker_obs_space for i in range(self.env.n_agents)},
                'p': planner_obs_space,
            }
            self.action_spaces = {
                **{str(i): worker_action_space for i in range(self.env.n_agents)},
                'p': planner_action_space,
            }
            # Expose spaces in RLlib's preferred multi-agent format.
            self.observation_space = SpaceDict(self.observation_spaces)
            self.action_space = SpaceDict(self.action_spaces)
            self._obs_space_in_preferred_format = True
            self._action_space_in_preferred_format = True

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.env.seed(seed)
            obs = self.env.reset()
            flat_obs = {
                agent_id: _policy_obs(agent_obs) for agent_id, agent_obs in obs.items()
            }
            infos = {agent_id: {} for agent_id in flat_obs}
            return flat_obs, infos

        def step(self, action_dict):
            formatted_actions = {}
            for i in range(self.env.n_agents):
                aid = str(i)
                action = action_dict.get(aid, 0)
                if isinstance(action, np.ndarray):
                    action = action.item()
                elif isinstance(action, (list, tuple)):
                    action = action[0]
                formatted_actions[aid] = int(action)

            planner_action = action_dict.get('p', [0] * len(planner_action_space.nvec))
            if isinstance(planner_action, np.ndarray):
                planner_action = planner_action.tolist()
            elif not isinstance(planner_action, (list, tuple)):
                planner_action = [planner_action]
            formatted_actions['p'] = [int(x) for x in planner_action]
            obs, rewards, done, info = self.env.step(formatted_actions)
            flat_obs = {agent_id: _policy_obs(agent_obs) for agent_id, agent_obs in obs.items()}
            terminateds = {agent_id: bool(done.get(agent_id, False)) for agent_id in flat_obs}
            terminateds['__all__'] = bool(done.get('__all__', False))
            truncateds = {agent_id: False for agent_id in flat_obs}
            truncateds['__all__'] = False
            infos = {agent_id: info.get(agent_id, {}) for agent_id in flat_obs}
            return flat_obs, rewards, terminateds, truncateds, infos

        def get_observation_space(self, agent_id):
            return planner_obs_space if agent_id == 'p' else worker_obs_space

        def get_action_space(self, agent_id):
            return planner_action_space if agent_id == 'p' else worker_action_space

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return 'planner' if agent_id == 'p' else 'worker'

    env_class = FoundationMultiAgentEnv
    env_config = dict(effective_env_config, seed=seed)

    # This callback logs what the AC planner reward would be for the current
    # env state. It does not replace the planner's training signal.
    from ray.rllib.algorithms.callbacks import DefaultCallbacks

    class ACCallbacks(DefaultCallbacks):
        """Logs Agency Calculus planner reward candidates after each episode."""

        def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
            # Record condition-specific reward diagnostics for later comparison.
            envs = base_env.get_sub_environments()
            for env_inst in envs:
                try:
                    raw_env = env_inst.env if hasattr(env_inst, 'env') else env_inst
                    planner_r = compute_planner_reward_from_env(raw_env, condition)
                    # Actual reward replacement requires env/scenario changes in
                    # the AI Economist fork; this repo only logs diagnostics.
                    episode.custom_metrics['planner_ac_reward'] = planner_r
                    episode.custom_metrics['floor_utility'] = min(extract_utilities(raw_env))
                except Exception:
                    pass  # Don't crash training on metric errors

    trainer_config = {
        'env': env_class,
        'env_config': env_config,
        # Keep training on the local worker for notebook/local runs so Ray
        # does not need to import this ad-hoc src module in remote actors.
        'num_workers': 0,
        'num_gpus': device_info['num_gpus'],
        'seed': seed,
        'train_batch_size': TRAIN_BATCH_SIZE,
        'rollout_fragment_length': ROLLOUT_FRAGMENT_LENGTH,
        'framework': 'torch',
        'callbacks': ACCallbacks,
        'lr': 3e-4,
        'gamma': 0.998,
        'lambda': 0.98,
        'clip_param': 0.2,
        'vf_clip_param': 10.0,
        'entropy_coeff': 0.025,
        'multiagent': {
            'policies': {
                'worker': (None, worker_obs_space, worker_action_space, {}),
                'planner': (None, planner_obs_space, planner_action_space, {}),
            },
            'policy_mapping_fn': policy_mapping_fn,
        },
    }

    trainer = PPO(config=trainer_config)
    env = make_env(seed=seed, env_config_overrides=effective_env_config)

    timesteps_trained = 0
    eval_count = 0
    start_time = time.time()

    while timesteps_trained < total_timesteps:
        result = trainer.train()
        timesteps_trained = result.get('timesteps_total', timesteps_trained + TRAIN_BATCH_SIZE)

        # Evaluate at intervals
        if timesteps_trained >= (eval_count + 1) * eval_interval:
            eval_count += 1

            # Get policies for eval
            try:
                policy_dict = {
                    'worker': trainer.get_policy('worker'),
                    'planner': trainer.get_policy('planner'),
                }
            except Exception:
                policy_dict = {}

            agg = run_eval_episodes(env, policy_dict, condition, n_episodes=20)
            logger.record_aggregated(step=timesteps_trained, agg=agg)

            if verbose:
                floor_u = agg.get('floor_utility', {}).get('mean', float('nan'))
                total_u = agg.get('total_utility', {}).get('mean', float('nan'))
                gini = agg.get('gini_wealth', {}).get('mean', float('nan'))
                elapsed = time.time() - start_time
                print(f'  Step {timesteps_trained:>10,}  '
                      f'floor_u={floor_u:.3f}  total_u={total_u:.3f}  '
                      f'gini={gini:.3f}  [{elapsed:.0f}s]')

    # Save results
    save_path = os.path.join(results_dir, f'{condition}_seed{seed}_metrics.npz')
    logger.save(save_path)

    trainer.stop()

    if verbose:
        elapsed = time.time() - start_time
        print(f'\nTraining complete in {elapsed:.0f}s')
        print(f'Saved to: {save_path}')

    return logger


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_condition(
    condition: str,
    seeds: Optional[List[int]] = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    results_dir: str = '../results',
) -> List[MetricsLogger]:
    """
    Run all seeds for one condition sequentially.

    Args:
        condition: Experiment condition name.
        seeds: List of seeds (default: [0, 1, 2, 3, 4])
        total_timesteps: Steps per seed
        results_dir: Output directory

    Returns:
        List of MetricsLogger, one per seed.
    """
    condition = validate_condition_name(condition)

    if seeds is None:
        seeds = list(range(N_SEEDS))

    loggers = []
    for seed in seeds:
        logger = run_training(
            condition=condition,
            seed=seed,
            total_timesteps=total_timesteps,
            results_dir=results_dir,
        )
        loggers.append(logger)

    return loggers
