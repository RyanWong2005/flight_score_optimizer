import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

def ground_run_dist_tw(ws: Array, p: dict[str, float]) -> Array:
    g = p["gravity_sea_level_m_per_s2"]
    rho = p["competition_density_kg_m3"]
    cl_TO = p["cl_ground_roll_unitless"]
    cl_max_TO = p["cl_max_TO_unitless"]
    cd_TO = p["cd_TO_unitless"]
    s_g = p["distance_ground_run_m"]
    mu = p["mu_TO_unitless"]

    return (1.21 / (g * rho * cl_max_TO * s_g)) * ws + (0.605 / cl_max_TO
                                                     ) * (cd_TO - mu * cl_TO) + mu