import numpy as np
import numpy.typing as npt

from aerodynamics.aero import lift_induced_drag_factor

Array = npt.NDArray[np.float64]

def level_turn_tw(
        ws: Array, 
        AR: Array, 
        k: Array, 
        q: Array, 
        p: dict[str, float],
    ) -> Array:

    k = lift_induced_drag_factor(AR, p)
    cd0 = p["cd_min_unitless"]
    n = p["load_factor_g"]

    return q * ((cd0 / ws) + k * ((n / q) ** 2) * ws)