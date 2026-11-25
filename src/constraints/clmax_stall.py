import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def max_lift_coefficient(AR: Array) -> Array:
    a = -0.001741420896367
    b = 0.056081348492323
    c = 0.841203061315264

    return a * AR**2 + b * AR + c


def stall_ws_max_clean(AR: Array, p: dict[str, float]) -> Array:
    return p["dynamic_pressure_stall_pa"] * max_lift_coefficient(AR)