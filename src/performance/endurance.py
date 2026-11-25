import numpy as np


MINUTES_PER_HOUR = 60.0


def calculate_endurance_hours(
    p: dict[str, float],
    weight_n: float,
    wing_area_m2: float,
    induced_drag_factor: float,
) -> float:
    """Calculate maximum endurance in hours for one design sweep."""
    eta_total = p["eta_total_unitless"]
    density = p["competition_density_kg_m3"]
    cd_min = p["cd_min_unitless"]
    battery_voltage = p["battery_voltage_v"]
    battery_capacity = p["battery_capacity_ah"]
    range_time_factor = p["range_time_factor_unitless"]
    peukert_constant = p["peukert_constant_unitless"]

    positive_values = {
        "total efficiency": eta_total,
        "air density": density,
        "minimum drag coefficient": cd_min,
        "battery voltage": battery_voltage,
        "battery capacity": battery_capacity,
        "range-time factor": range_time_factor,
        "Peukert constant": peukert_constant,
        "aircraft weight": weight_n,
        "wing area": wing_area_m2,
        "induced-drag factor": induced_drag_factor,
    }
    if any(
        not np.isfinite(value) or value <= 0
        for value in positive_values.values()
    ):
        invalid_values = ", ".join(
            name
            for name, value in positive_values.items()
            if not np.isfinite(value) or value <= 0
        )
        raise ValueError(
            f"Endurance inputs must be positive and finite: {invalid_values}."
        )

    denominator = (
        (2 / np.sqrt(density * wing_area_m2))
        * np.power(cd_min, 0.25)
        * np.power(
            2 * weight_n * np.sqrt(induced_drag_factor / 3),
            1.5,
        )
    )
    usable_battery_energy_wh = (
        eta_total * battery_voltage * battery_capacity
    )

    return float(
        np.power(range_time_factor, 1 - peukert_constant)
        * np.power(
            usable_battery_energy_wh / denominator,
            peukert_constant,
        )
    )


def endurance_hours_to_minutes(endurance_hours: float) -> float:
    """Convert endurance in hours to minutes."""
    return endurance_hours * MINUTES_PER_HOUR