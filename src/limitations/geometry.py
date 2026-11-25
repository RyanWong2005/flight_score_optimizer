MAX_WINGSPAN_M = 96 * 0.0254


def wingspan_is_feasible(span_m: float) -> bool:
    return span_m < MAX_WINGSPAN_M