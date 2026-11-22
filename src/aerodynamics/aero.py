import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

def oswald_efficiency(AR: Array, p: dict[str, float]) -> Array:

    # ------------------------------------------------------------ #
    # Lambda Function #
    # ------------------------------------------------------------ #
    a = 0.0524
    b = 0.15
    c = 0.1659
    d = 0.0706
    e = 0.0119

    lbda = p["taper_ratio_unitless"]

    lambda_aero = (
        a * lbda ** 4
        - b * lbda ** 3
        + c * lbda ** 2
        - d * lbda
        + e
    )

    # ------------------------------------------------------------ #
    # Oswald Efficiency #
    # ------------------------------------------------------------ #

    oswald_efficiency_unitless = 1.0 / (1.0 + AR * lambda_aero)

    return oswald_efficiency_unitless

def lift_induced_drag_factor(AR: Array, p: dict[str, float]) -> Array:

    e = oswald_efficiency(AR, p)

    return 1.0 / (np.pi * AR * e)

def lift_to_drag_max(AR: Array, p: dict[str, float]) -> Array:

    e = oswald_efficiency(AR, p)
    Cd0 = p["cd_min_unitless"]

    return 0.5 * np.sqrt((np.pi * AR * e) / Cd0)

def max_lift_coefficient(AR: Array) -> Array:

    # ------------------------------------------------------------ #
    # Cl,max Function #
    # ------------------------------------------------------------ #
    a = 0.001741420896367
    b = 0.056081348492323
    c = 0.841203061315264

    return -(a * AR ** 2) + (b * AR) + c

def dynamic_pressure(p:dict[str, float]) -> dict[str, float]:
    
    p = dict(p)
    rho = p["competition_density_kg_m3"]
    p["dynamic_pressure_stall_pa"] = 0.5 * rho * p["target_stall_speed_m_per_s"] ** 2
    p["dynamic_pressure_cruise_pa"] = 0.5 * rho * p["target_cruise_speed_m_per_s"] ** 2

    return p

def dynamic_pressure_from_velocity(rho: float, V: Array) -> Array:
    return 0.5 * rho * np.asarray(V) ** 2