"""
Plotting utilities for Agency Calculus empirical validation.

Standard comparison plots across SUM / NASH / JAM conditions.
Designed to work in Jupyter notebooks (Colab/Kaggle).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

CONDITION_COLORS = {
    "sum": "#e74c3c",   # red
    "nash": "#f39c12",  # orange
    "jam": "#2ecc71",   # green
    # Limitation variants
    "jam_epsilon": "#27ae60",
    "jam_softmin": "#1abc9c",
}

CONDITION_LABELS = {
    "sum": "SUM (baseline)",
    "nash": "NASH",
    "jam": "JAM",
    "jam_epsilon": "JAM + ε (limitation)",
    "jam_softmin": "JAM + softmin (limitation)",
}

METRIC_LABELS = {
    "floor_utility": "Floor Agent Utility  min(u_i)",
    "floor_agency": "Floor Agent Agency  min(A_i)",
    "gini_wealth": "Gini Coefficient (Wealth)",
    "resource_concentration": "Resource Concentration  max/mean",
    "floor_tax_rate": "Tax Rate on Floor Agent",
    "floor_action_diversity": "Floor Agent Action Diversity",
    "total_utility": "Total System Utility  Σu_i",
    "fhi": "Floor Harm Index (FHI)",
    "mean_utility": "Mean Utility",
    "mean_agency": "Mean Agency",
}

METRIC_HIGHER_IS_BETTER = {
    "floor_utility": True,
    "floor_agency": True,
    "gini_wealth": False,
    "resource_concentration": False,
    "floor_tax_rate": False,
    "floor_action_diversity": True,
    "total_utility": True,
    "fhi": False,
    "mean_utility": True,
    "mean_agency": True,
}


def _set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    })


# ---------------------------------------------------------------------------
# Single-metric training curve
# ---------------------------------------------------------------------------

def plot_training_curve(
    data: Dict[str, Dict[str, np.ndarray]],
    metric: str,
    ax: Optional[plt.Axes] = None,
    shade_std: bool = True,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot a single metric over training steps for all conditions.

    Args:
        data: Nested dict: condition_name -> {"steps": array, metric: array,
              metric+"_std": array (optional)}
        metric: Metric key to plot.
        ax: Matplotlib axes to plot on (creates new figure if None).
        shade_std: Whether to shade ±1 std across seeds.
        title: Plot title (defaults to metric label).

    Returns:
        The axes object.
    """
    _set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    for condition, arrays in data.items():
        if metric not in arrays:
            continue
        steps = arrays.get("steps", arrays.get("step", np.arange(len(arrays[metric]))))
        mean = arrays[metric]
        color = CONDITION_COLORS.get(condition, "gray")
        label = CONDITION_LABELS.get(condition, condition)

        ax.plot(steps, mean, color=color, label=label, linewidth=2)

        std_key = f"{metric}_std"
        if shade_std and std_key in arrays:
            std = arrays[std_key]
            ax.fill_between(steps, mean - std, mean + std,
                            color=color, alpha=0.15)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or METRIC_LABELS.get(metric, metric))
    ax.legend(framealpha=0.9)
    return ax


# ---------------------------------------------------------------------------
# Dashboard: key metrics side-by-side
# ---------------------------------------------------------------------------

def plot_main_dashboard(
    data: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    6-panel dashboard of the main metrics for the SUM/NASH/JAM comparison.

    Panels:
      1. Floor utility        (critical result)
      2. Total utility        (tradeoff)
      3. Gini wealth          (inequality)
      4. Floor agency (POLI)  (agency floor)
      5. Tax on floor agent   (extraction)
      6. FHI                  (floor isolation)

    Args:
        data: condition_name -> {"steps": ..., metric_name: ..., ...}
        save_path: If provided, save figure to this path.
    """
    _set_style()
    metrics = [
        "floor_utility", "total_utility",
        "gini_wealth", "floor_agency",
        "floor_tax_rate", "fhi",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        plot_training_curve(data, metric, ax=ax)

    # Add legend to first panel only, remove from others
    for ax in axes[1:]:
        ax.get_legend().remove()
    handles = [
        mpatches.Patch(color=CONDITION_COLORS[c], label=CONDITION_LABELS[c])
        for c in ["sum", "nash", "jam"] if c in data
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=12)

    fig.suptitle(
        "Agency Calculus Validation: SUM vs NASH vs JAM\n"
        "AI Economist Multi-Agent Economic Simulation",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved dashboard to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Final-value bar chart
# ---------------------------------------------------------------------------

def plot_final_values_bar(
    final_metrics: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing final (end-of-training) metric values per condition.

    Args:
        final_metrics: condition_name -> {metric_name: mean_value, ...}
        metrics: Which metrics to include (default: main 6).
        save_path: Optional save path.
    """
    _set_style()
    if metrics is None:
        metrics = ["floor_utility", "total_utility", "gini_wealth",
                   "floor_agency", "floor_tax_rate", "fhi"]

    conditions = list(final_metrics.keys())
    n_metrics = len(metrics)
    n_conds = len(conditions)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = [final_metrics[c].get(metric, 0) for c in conditions]
        colors = [CONDITION_COLORS.get(c, "gray") for c in conditions]
        bars = ax.bar(conditions, values, color=colors, edgecolor="white",
                      linewidth=1.5, width=0.6)

        # Annotate bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.set_ylabel("")
        ax.set_xticklabels(
            [CONDITION_LABELS.get(c, c) for c in conditions],
            rotation=20, ha="right", fontsize=9,
        )

    fig.suptitle("Final Metric Values by Condition (mean over seeds)",
                 fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved bar chart to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# POLI decomposition radar chart
# ---------------------------------------------------------------------------

def plot_poli_radar(
    poli_means: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Radar/spider chart of POLI dimension means per condition.

    Args:
        poli_means: condition -> {dimension: mean_value}
                    Dimensions: prerequisites, options, levers, impact, knowledge
        save_path: Optional save path.
    """
    _set_style()
    dimensions = ["prerequisites", "options", "levers", "impact", "knowledge"]
    n_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7),
                            subplot_kw=dict(polar=True))

    for condition, poli in poli_means.items():
        values = [poli.get(d, 0) for d in dimensions]
        values += values[:1]
        color = CONDITION_COLORS.get(condition, "gray")
        label = CONDITION_LABELS.get(condition, condition)
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.capitalize() for d in dimensions], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("POLI Agency Dimensions by Condition\n(floor agent, end of training)",
                 pad=20, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved radar chart to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Limitation experiment: degradation curves
# ---------------------------------------------------------------------------

def plot_limitation_comparison(
    data: Dict[str, Dict[str, np.ndarray]],
    metric: str = "floor_utility",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare JAM vs JAM+epsilon vs JAM+softmin on a single metric.

    Args:
        data: condition_name -> {"steps": ..., metric: ..., ...}
        metric: Which metric to show (default: floor_utility).
        title: Plot title.
        save_path: Optional save path.
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_training_curve(data, metric, ax=ax,
                        title=title or f"Limitation Experiment: {METRIC_LABELS.get(metric, metric)}")

    ax.set_title(
        title or (
            f"Limitation Experiment: {METRIC_LABELS.get(metric, metric)}\n"
            "Epsilon and softmin degrade the floor guarantee"
        )
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved limitation plot to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Utility: load and merge seed results
# ---------------------------------------------------------------------------

def merge_seed_results(
    seed_loggers: list,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge per-seed MetricsLogger objects into mean ± std arrays.

    Args:
        seed_loggers: List of MetricsLogger instances (one per seed).
        metric: Metric key to aggregate.

    Returns:
        (steps, mean, std) as numpy arrays, aligned to the shortest run.
    """
    all_steps = [logger.to_arrays().get("step", np.array([])) for logger in seed_loggers]
    all_vals = [logger.to_arrays().get(metric, np.array([])) for logger in seed_loggers]

    min_len = min(len(v) for v in all_vals if len(v) > 0)
    if min_len == 0:
        return np.array([]), np.array([]), np.array([])

    steps = all_steps[0][:min_len]
    matrix = np.stack([v[:min_len] for v in all_vals])
    return steps, matrix.mean(axis=0), matrix.std(axis=0)
