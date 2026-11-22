def stall_ws_max(p: dict[str, float]) -> float:
    return p["dynamic_pressure_stall_pa"] * p["cl_max_LDG_unitless"]