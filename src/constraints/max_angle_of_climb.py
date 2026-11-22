import numpy as np 
import numpy.typing as npt

from aerodynamics.aero import lift_induced_drag_factor

Array = npt.NDArray[np.float64]

def max_aoc_tw(ws: Array, AR: Array, k: Array, p: dict[str, float]) -> Array:

    k = lift_induced_drag_factor(AR, p)
    gamma = np.radians(p["climb_angle_degree"])
    cd0 = p["cd_min_unitless"]

    tw = np.sin(gamma) + 2.0 * np.cos(gamma) * np.sqrt(k * cd0)

    return np.broadcast_to(
        tw,
        np.broadcast_shapes(np.shape(tw), np.shape(ws))
    )