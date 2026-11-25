
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
) -> None:

    # ------------------------------------------------------------ #
    # Initialize Plot
    # ------------------------------------------------------------ #

    fig, ax = plt.subplots(figsize=(8, 5))

    # ------------------------------------------------------------ #
    # P/W Constraint Curves
    # ------------------------------------------------------------ #

    for name, y in pw_curves.items():
        ax.plot(ws, y, label=name)

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

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ------------------------------------------------------------ #
    # Save Figure
    # ------------------------------------------------------------ #

    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)