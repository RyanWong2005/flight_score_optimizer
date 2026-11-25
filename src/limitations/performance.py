MAX_GROSS_WEIGHT_KG = 24.94331066
MIN_CRUISE_TO_STALL_SPEED_RATIO = 1.3


def gross_weight_is_feasible(gross_weight_kg: float) -> bool:
    return gross_weight_kg < MAX_GROSS_WEIGHT_KG


def cruise_stall_speed_is_feasible(
    cruise_speed_m_s: float,
    stall_speed_m_s: float,
) -> bool:
    return cruise_speed_m_s >= (
        MIN_CRUISE_TO_STALL_SPEED_RATIO * stall_speed_m_s
    )