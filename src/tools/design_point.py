
import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DesignPoint:
    index: int
    ws: float
    pw: float
    governing: str
    ws_limit: float
    limited_by: str
    at_limit: bool
    pw_min: float
    ws_at_pw_min: float


def design_point(
    ws: Array,
    pw_curves: dict[str, Array],
    ws_limits: dict[str, float],
    ws_min_limits: dict[str, float] | None = None,
    *,
    strategy: str,
) -> DesignPoint:

    # ------------------------------------------------------------ #
    # Input Validation
    # ------------------------------------------------------------ #

    if strategy not in {"max_ws", "min_pw"}:
        raise ValueError(f"Unknown design-point strategy: {strategy}")

    if not pw_curves:
        raise ValueError("No P/W constraint curves provided.")

    if not ws_limits:
        raise ValueError("No W/S upper limits provided.")

    names = list(pw_curves)

    if any(curve.shape != ws.shape for curve in pw_curves.values()):
        raise ValueError("All P/W curves must have the same shape as W/S.")

    # ------------------------------------------------------------ #
    # Constraint Envelope
    # ------------------------------------------------------------ #

    stack = np.vstack([pw_curves[n] for n in names])
    envelope = np.fmax.reduce(stack, axis=0)

    # ------------------------------------------------------------ #
    # W/S Limits
    # ------------------------------------------------------------ #

    if any(np.isnan(v) for v in ws_limits.values()):
        raise ValueError(f"A W/S limit is NaN: {ws_limits}")

    limited_by = min(ws_limits, key=lambda n: ws_limits[n])
    ws_limit = ws_limits[limited_by]

    ws_floor = max(ws_min_limits.values(), default=0.0) if ws_min_limits else 0.0

    if ws_floor > ws_limit:
        raise ValueError(
            f"Infeasible: W/S floor {ws_floor:.1f} "
            f"> ceiling {ws_limit:.1f} ({limited_by})"
        )

    if ws_limit > ws[-1]:
        warnings.warn(
            f"W/S range ends at {ws[-1]} before the first limit "
            f"({ws_limit:.1f}); extend the range."
        )

    # ------------------------------------------------------------ #
    # Feasible Region
    # ------------------------------------------------------------ #

    feasible = (
        (ws >= ws_floor)
        & (ws <= ws_limit)
        & ~np.isnan(envelope)
    )

    if not feasible.any():
        raise ValueError("No feasible W/S in the window")

    idx = np.flatnonzero(feasible)

    # ------------------------------------------------------------ #
    # Design Point Selection
    # ------------------------------------------------------------ #

    i_min = int(idx[np.argmin(envelope[idx])])

    i = int(idx[-1]) if strategy == "max_ws" else i_min

    governing = names[int(np.nanargmax(stack[:, i]))]

    at_limit = i + 1 < ws.size and bool(ws[i + 1] > ws_limit)

    return DesignPoint(
        i,
        float(ws[i]),
        float(envelope[i]),
        governing,
        ws_limit,
        limited_by,
        at_limit,
        float(envelope[i_min]),
        float(ws[i_min])
    )
