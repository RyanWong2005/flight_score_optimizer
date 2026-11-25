MAX_GROSS_WEIGHT_KG = 24.94331066
MIN_CRUISE_TO_STALL_SPEED_RATIO = 1.3
AVERAGE_ENDURANCE_MIN = 1.5
ENDURANCE_MULTIPLIER = 1.5
MIN_ENDURANCE_MIN = AVERAGE_ENDURANCE_MIN * ENDURANCE_MULTIPLIER


def gross_weight_is_feasible(gross_weight_kg: float) -> bool:
    return gross_weight_kg < MAX_GROSS_WEIGHT_KG


def cruise_stall_speed_is_feasible(
    cruise_speed_m_s: float,
    stall_speed_m_s: float,
) -> bool:
    return cruise_speed_m_s >= (
        MIN_CRUISE_TO_STALL_SPEED_RATIO * stall_speed_m_s
    )


def endurance_is_feasible(endurance_min: float) -> bool:
    return endurance_min >= MIN_ENDURANCE_MIN