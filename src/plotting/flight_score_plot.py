from pathlib import Path
from typing import cast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.axes3d import Axes3D


def _validate_grids(
    AR_grid,
    V_grid,
    feasible_grid,
    *value_grids,
):
    grids = (
        np.asarray(AR_grid),
        np.asarray(V_grid),
        np.asarray(feasible_grid, dtype=bool),
        *(np.asarray(grid) for grid in value_grids),
    )

    if any(grid.shape != grids[0].shape for grid in grids[1:]):
        raise ValueError("All input grids must have identical shapes.")

    return grids


def _save_figure(fig, save_path):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )


def _display_figure(fig, show):
    try:
        if show:
            plt.show(block=True)
    finally:
        plt.close(fig)


# =============================================================
# FLIGHT SCORE SURFACE
# =============================================================

def plot_score_surface(
    AR_grid,
    V_grid,
    score_grid,
    feasible_grid,
    save_path=None,
    show=False,
):

    AR_grid, V_grid, feasible_grid, score_grid = _validate_grids(
        AR_grid,
        V_grid,
        feasible_grid,
        score_grid,
    )

    score_plot = np.ma.masked_invalid(
        np.ma.masked_where(~feasible_grid, score_grid)
    )

    fig = plt.figure(figsize=(11, 8))

    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.95)

    surface = ax.plot_surface(
        AR_grid,
        V_grid,
        score_plot,
        cmap="viridis",
        edgecolor="none",
        alpha=0.9,
    )

    fig.colorbar(
        surface,
        ax=ax,
        shrink=0.65,
        label="Flight Score",
    )

    # ---------------------------------------------------------
    # Mark highest-scoring feasible designs
    # ---------------------------------------------------------

    feasible_scores = np.where(feasible_grid, score_grid, np.nan)

    if np.any(np.isfinite(feasible_scores)):

        best_mask = (
            np.isfinite(feasible_scores)
            & np.isclose(feasible_scores, np.nanmax(feasible_scores))
        )

        ax.plot(
            AR_grid[best_mask],
            V_grid[best_mask],
            score_grid[best_mask],
            linestyle="None",
            color="red",
            marker="*",
            markersize=10,
            label="Best score",
        )

        ax.legend()

    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Cruise Velocity [m/s]")
    ax.set_zlabel("Flight Score")

    ax.set_title(
        "Aircraft Flight Score Optimization"
    )

    _save_figure(fig, save_path)

    _display_figure(fig, show)


# =============================================================
# FLIGHT SCORE SURFACES (MULTIPLE BOTTLE CONFIGURATIONS)
# =============================================================

# Ordinal blue ramp (dark -> light) used to color configurations
# in order of increasing full-bottle count, so the lowest-scoring surface
# is the darkest and the upper surfaces stay light enough to see through.
_ORDINAL_CMAP = LinearSegmentedColormap.from_list(
    "ordinal_blue",
    [
        "#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf",
        "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef",
    ],
)


def _ordinal_colors(n):
    # Sample the continuous ramp so any number of configurations
    # gets distinct, evenly spaced shades.
    if n == 1:
        return [_ORDINAL_CMAP(0.0)]

    return [_ORDINAL_CMAP(x) for x in np.linspace(0.0, 1.0, n)]


def plot_score_surfaces(
        AR_grid,
        V_grid,
        score_grids,
        feasible_grid,
        labels,
        save_path=None,
        show=False,
):
    """
    Plot one flight-score surface per configuration on a shared
    3D axis.

    score_grids : array of shape (n_config, n_ar, n_v)
    labels      : one legend label per configuration
    """

    score_grids = np.asarray(score_grids, dtype=float)

    if score_grids.ndim != 3:
        raise ValueError("score_grids must have shape (n_config, n_ar, n_v).")

    if len(labels) != score_grids.shape[0]:
        raise ValueError("One label is required per score grid.")

    AR_grid, V_grid, feasible_grid = _validate_grids(
        AR_grid, 
        V_grid,
        feasible_grid,
        *score_grids,
    )[:3]

    colors = _ordinal_colors(score_grids.shape[0])

    fig = plt.figure(figsize=(11, 8))

    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
    fig.subplots_adjust(left=0.05, right=0.80, bottom=0.05, top=0.95)

    legend_handles = []

    for score_grid, color, label in zip(score_grids, colors, labels):

        score_plot = np.ma.masked_invalid(
            np.ma.masked_where(~feasible_grid, score_grid)
        )

        ax.plot_surface(
            AR_grid,
            V_grid,
            score_plot,
            color=color,
            edgecolor="none",
            alpha=0.85,
            shade=False
        )

        legend_handles.append(
            Patch(facecolor=color, label=label)
        )

    # =============================================================
    # Mark highest-scoring feasible designs (all configurations)
    # =============================================================

    feasible_scores = np.where(feasible_grid, score_grids, np.nan)

    if np.any(np.isfinite(feasible_scores)):

        best_mask = (
            np.isfinite(feasible_scores)
            & np.isclose(feasible_scores, np.nanmax(feasible_scores))
        )

        best_config, best_ar, best_v = np.nonzero(best_mask)

        best_line, = ax.plot(
            AR_grid[best_ar, best_v],
            V_grid[best_ar, best_v],
            score_grids[best_config, best_ar, best_v],
            linestyle="None",
            color="red",
            marker="*",
            markersize=10,
            label="Best score",
        )

        legend_handles.append(best_line)

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )

    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Cruise Velocity [m/s]")
    ax.set_zlabel("Flight score")

    ax.set_title(
        "Aircraft Flight Score Optimization by Bottle Configuration"
    )

    _save_figure(fig, save_path)

    _display_figure(fig, show)
    

# =============================================================
# DESIGN FEASIBILITY
# =============================================================

def plot_feasibility(
    AR_grid,
    V_grid,
    feasible_grid,
    save_path=None,
    show=False,
):

    AR_grid, V_grid, feasible_grid = _validate_grids(
        AR_grid,
        V_grid,
        feasible_grid,
    )

    fig = plt.figure(figsize=(11, 8))

    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    # ---------------------------------------------------------
    # Feasible designs
    # ---------------------------------------------------------

    ax.scatter(
        AR_grid[feasible_grid],
        V_grid[feasible_grid],
        1,
        color="green",
        marker="o",
        s=20,
        label="Feasible",
    )

    # ---------------------------------------------------------
    # Infeasible designs
    # ---------------------------------------------------------

    infeasible_grid = ~feasible_grid

    ax.scatter(
        AR_grid[infeasible_grid],
        V_grid[infeasible_grid],
        0,
        color="red",
        marker="x",
        s=20,
        label="Infeasible",
    )

    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Cruise Velocity [m/s]")
    ax.set_zlabel("Feasibility")

    ax.zaxis.set_ticks([0, 1])
    ax.zaxis.set_ticklabels([
        "Infeasible",
        "Feasible",
    ])

    ax.set_title(
        "Aircraft Design Feasibility"
    )

    ax.legend()

    _save_figure(fig, save_path)

    _display_figure(fig, show)


# =============================================================
# FLIGHT SCORE CONTOUR
# =============================================================

def plot_score_contour(
    AR_grid,
    V_grid,
    score_grid,
    feasible_grid,
    save_path=None,
    show=False,
):

    AR_grid, V_grid, feasible_grid, score_grid = _validate_grids(
        AR_grid,
        V_grid,
        feasible_grid,
        score_grid,
    )

    score_plot = np.ma.masked_invalid(
        np.ma.masked_where(~feasible_grid, score_grid)
    )

    fig, ax = plt.subplots(
        figsize=(10, 7),
    )

    contour = ax.contourf(
        AR_grid,
        V_grid,
        score_plot,
        levels=20,
        cmap="viridis",
    )

    fig.colorbar(
        contour,
        ax=ax,
        label="Flight Score",
    )

    feasible_scores = np.where(feasible_grid, score_grid, np.nan)

    if np.any(np.isfinite(feasible_scores)):

        best_mask = (
            np.isfinite(feasible_scores)
            & np.isclose(feasible_scores, np.nanmax(feasible_scores))
        )

        ax.scatter(
            AR_grid[best_mask],
            V_grid[best_mask],
            color="red",
            marker="*",
            s=100,
            label="Best score",
        )

        ax.legend()

    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Cruise Velocity [m/s]")

    ax.set_title(
        "Aircraft Flight Score Contour"
    )

    _save_figure(fig, save_path)

    _display_figure(fig, show)


# =============================================================
# CRUISE POWER SURFACE
# =============================================================

def plot_metric_surface(
    AR_grid,
    V_grid,
    metric_grid,
    feasible_grid,
    title,
    z_label,
    save_path=None,
    show=False,
):

    AR_grid, V_grid, feasible_grid, metric_grid = _validate_grids(
        AR_grid,
        V_grid,
        feasible_grid,
        metric_grid,
    )

    metric_plot = np.ma.masked_invalid(
        np.ma.masked_where(~feasible_grid, metric_grid)
    )

    fig = plt.figure(figsize=(11, 8))

    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.95)

    surface = ax.plot_surface(
        AR_grid,
        V_grid,
        metric_plot,
        cmap="plasma",
        edgecolor="none",
        alpha=0.9,
    )

    fig.colorbar(
        surface,
        ax=ax,
        shrink=0.65,
        label=z_label,
    )

    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Cruise Velocity [m/s]")
    ax.set_zlabel(z_label)

    ax.set_title(title)

    _save_figure(fig, save_path)

    _display_figure(fig, show)