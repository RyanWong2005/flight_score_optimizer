import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

G = 9.80665

def tw_to_pw(tw: Array, v: float | Array, eta: float) -> Array:
    return tw * v / eta

def speed_at_cl(ws: Array, cl: float | Array, rho: float) -> Array:
    return np.sqrt(2.0 * ws / (rho * cl))

def pw_per_n_to_per_kg(pw: Array) -> Array:
    return pw * G