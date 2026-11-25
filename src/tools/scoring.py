
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BottleResult:

    full_bottles: int
    empty_bottles: int

    full_weight: float
    empty_weight: float

    total_payload_used: float
    remaining_payload: float

    flight_score: int


def optimize_bottles(
    payload_weight: float,

    full_bottle_weight: float = 1.81437,
    empty_bottle_weight: float = 0.453592,

    full_bottle_score: int = 11,
    empty_bottle_score: int = 3,

    max_bottles: int | None = None,
) -> BottleResult:

    """
    Integer optimization of full and empty bottle quantities.

    Objective:
        Maximize full_bottle_score*N_full + empty_bottle_score*N_empty
        (default: 11*N_full + 3*N_empty)

    Constraints:
        full_bottle_weight*N_full + empty_bottle_weight*N_empty
            <= payload_weight
        (default: 1.81437*N_full + 0.453592*N_empty, in kg)

        N_full + N_empty <= max_bottles (if specified)

        N_full, N_empty are non-negative integers.
    """

    if not np.isfinite(payload_weight) or payload_weight < 0:
        raise ValueError("Invalid payload weight.")

    if full_bottle_weight <= 0 or empty_bottle_weight <= 0:
        raise ValueError("Bottle weights must be positive.")

    if max_bottles is not None and max_bottles < 0:
        raise ValueError("Maximum bottle count cannot be negative.")

    # Maximum possible full bottles based on payload weight.

    max_full = int(payload_weight // full_bottle_weight)

    if max_bottles is not None:
        max_full = min(max_full, max_bottles)

    best_score = -1

    best_full = 0
    best_empty = 0

    # Enumerate every possible full bottle count.

    for n_full in range(max_full + 1):

        remaining = (
            payload_weight
            - n_full * full_bottle_weight
        )

        # Maximum empty bottles for this full bottle count.

        n_empty = int(
            remaining // empty_bottle_weight
        )

        if max_bottles is not None:

            n_empty = min(
                n_empty,
                max_bottles - n_full
            )

        score = (
            n_full * full_bottle_score
            + n_empty * empty_bottle_score
        )

        if score > best_score:

            best_score = score

            best_full = n_full
            best_empty = n_empty

    # Calculate final payload utilization.

    full_weight = best_full * full_bottle_weight

    empty_weight = best_empty * empty_bottle_weight

    total_used = full_weight + empty_weight

    return BottleResult(

        full_bottles=best_full,
        empty_bottles=best_empty,

        full_weight=full_weight,
        empty_weight=empty_weight,

        total_payload_used=total_used,

        remaining_payload=payload_weight - total_used,

        flight_score=best_score,
    )
