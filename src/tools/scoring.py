
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ScoringResult:

    payload_weight: float

    full_bottles: int
    empty_bottles: int

    full_weight: float
    empty_weight: float

    flight_score: float


def optimize_bottles(
    payload_weight: float,
    full_bottle_weight: float = 2.1,
    empty_bottle_weight: float = 0.05,
    full_bottle_score: int = 11,
    empty_bottle_score: int = 3,
    max_full_bottles: int | None = None,
    max_empty_bottles: int | None = None,
) -> ScoringResult:

    """
    Determine the optimal integer combination of full and
    empty bottles that maximizes flight score.

    Constraint:
        W_full + W_empty <= payload_weight
    """

    if payload_weight < 0:
        raise ValueError("Payload weight cannot be negative.")

    if full_bottle_weight <= 0 or empty_bottle_weight <= 0:
        raise ValueError("Bottle weights must be positive.")

    max_full = int(payload_weight // full_bottle_weight)

    if max_full_bottles is not None:
        max_full = min(max_full, max_full_bottles)

    best_score = -1
    best_full = 0
    best_empty = 0

    for n_full in range(max_full + 1):

        remaining_weight = (
            payload_weight - n_full * full_bottle_weight
        )

        n_empty = int(
            remaining_weight // empty_bottle_weight
        )

        if max_empty_bottles is not None:
            n_empty = min(n_empty, max_empty_bottles)

        score = (
            full_bottle_score * n_full
            + empty_bottle_score * n_empty
        )

        if score > best_score:

            best_score = score
            best_full = n_full
            best_empty = n_empty

    full_weight = best_full * full_bottle_weight
    empty_weight = best_empty * empty_bottle_weight

    return ScoringResult(
        payload_weight=payload_weight,
        full_bottles=best_full,
        empty_bottles=best_empty,
        full_weight=full_weight,
        empty_weight=empty_weight,
        flight_score=best_score,
    )


def optimize_bottles_grid(
    payload_grid: np.ndarray,
    **kwargs,
):

    """
    Apply bottle optimization to an entire payload array.
    """

    payload_grid = np.asarray(payload_grid, dtype=float)

    score_grid = np.full(payload_grid.shape, np.nan)
    full_grid = np.full(payload_grid.shape, np.nan)
    empty_grid = np.full(payload_grid.shape, np.nan)

    for index in np.ndindex(payload_grid.shape):

        payload = payload_grid[index]

        if not np.isfinite(payload) or payload < 0:
            continue

        result = optimize_bottles(
            payload_weight=float(payload),
            **kwargs
        )

        score_grid[index] = result.flight_score
        full_grid[index] = result.full_bottles
        empty_grid[index] = result.empty_bottles

    return {
        "score": score_grid,
        "full_bottles": full_grid,
        "empty_bottles": empty_grid,
    }
