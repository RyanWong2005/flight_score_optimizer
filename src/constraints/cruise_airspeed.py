import numpy as np
import numpy.typing as npt

from aerodynamics.aero import lift_induced_drag_factor

Array = npt.NDArray[np.float64]

def cruise_speed_tw(
        ws: Array, 
        AR: Array, 
        k: Array,
        q: Array,
        p: dict[str, float],
    ) -> Array:

    k = lift_induced_drag_factor(AR, p)
    cd0 = p["cd_min_unitless"]

    return q * cd0 * (1 / ws) + k * (1 / q) * ws