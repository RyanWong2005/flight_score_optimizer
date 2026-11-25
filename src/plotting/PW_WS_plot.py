
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from tools.design_point import DesignPoint

Array = npt.NDArray[np.float64]


def plot_baseline(
    ws: Array,
    pw_curves: dict[str, Array],
    ws_limits: dict[str, float],
    dp: DesignPoint,
    path: Path,
    *,
    ar: float,
    velocity: float,
    cl_max_required: Array,
    cl_max_available: float,
    cl_max_required_at_stall_limit: float,
    target_stall_speed: float,
    close: bool = True,
) -> None:

    # ------------------------------------------------------------ #
    # Initialize Plot
    # ------------------------------------------------------------ #

    fig, ax = plt.subplots(figsize=(8, 5))
    cl_ax = ax.twinx()

    # ------------------------------------------------------------ #
    # P/W Constraint Curves
    # ------------------------------------------------------------ #

    for name, y in pw_curves.items():
        ax.plot(ws, y, label=name)

    cl_ax.plot(
        ws,
        cl_max_required,
        color="tab:purple",
        linestyle="-.",
        label=f"Required CL,max at {target_stall_speed:.1f} m/s",
    )
    cl_ax.axhline(
        cl_max_available,
        color="tab:green",
        linestyle=":",
        label=f"Clean CL,max at AR = {ar:.2f}",
    )
    stall_ws_limit = ws_limits["stall"]
    if ws[0] <= stall_ws_limit <= ws[-1]:
        cl_ax.hlines(
            cl_max_required_at_stall_limit,
            xmin=stall_ws_limit,
            xmax=ws[-1],
            color="tab:red",
            linestyle="--",
            label="Required CL,max at stall W/S limit",
        )
        cl_ax.plot(
            stall_ws_limit,
            cl_max_required_at_stall_limit,
            marker="o",
            color="tab:red",
            markersize=4,
        )

    # ------------------------------------------------------------ #
    # W/S Limits
    # ------------------------------------------------------------ #

    for name, x in ws_limits.items():
        ax.axvline(x, linestyle="--", color="gray")
        ax.text(
            x,
            0.02,
            f" {name}",
            rotation=90,
            transform=ax.get_xaxis_transform()
        )

    # ------------------------------------------------------------ #
    # Design Point
    # ------------------------------------------------------------ #

    ax.plot(
        dp.ws,
        dp.pw,
        "ko",
        label=f"Design point ({dp.ws:.1f}, {dp.pw:.1f})"
    )

    # ------------------------------------------------------------ #
    # Plot Formatting
    # ------------------------------------------------------------ #

    ax.set_title(f"AR = {ar:.2f}, V = {velocity:.2f} m/s")

    ax.set_ylim(0, 2 * dp.pw)

    ax.set_xlabel("W/S [N/m²]")
    ax.set_ylabel("P/W [W/kg]")
    cl_ax.set_ylabel("Required CL,max [-]")

    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    cl_handles, cl_labels = cl_ax.get_legend_handles_labels()
    ax.legend(
        handles + cl_handles,
        labels + cl_labels,
        fontsize=8,
    )

    # ------------------------------------------------------------ #
    # Save Figure
    # ------------------------------------------------------------ #

    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    if close:
        plt.close(fig)