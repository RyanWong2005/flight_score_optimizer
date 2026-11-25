
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import root_scalar


Array = npt.NDArray[np.float64]


# Historical regression constants:
# ln(We) = a * ln(W0) + b

CONSTANT_A = 0.4387
CONSTANT_B = 0.8347

# ---------------------------------------------------------
# Gross weight estimation based on payload
# ---------------------------------------------------------

@dataclass(frozen=True)
class EmptyWeightResult:
    gross_weight: float
    empty_weight: float
    empty_weight_fraction: float
    payload_weight: float
    payload_fraction: float
    battery_weight: float
    battery_fraction: float
    iterations: int
    converged: bool


def estimate_gross_weight(
    payload_weight: float,
    battery_weight: Optional[float] = None,
    battery_fraction: Optional[float] = None,
    a: float = CONSTANT_A,
    b: float = CONSTANT_B,
    initial_guess: float = 15.0,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> EmptyWeightResult:
    """
    Estimate aircraft gross weight using a historical empty-weight
    regression and an iterative weight-balance solution.

    Historical regression:
        ln(We) = a * ln(W0) + b

    Weight balance:
        W0 = Wpayload + Wbattery + We

    Battery input:
        Either battery_weight (kg) OR battery_fraction (Wbattery/W0).
        Exactly one must be provided.

    Parameters
    ----------
    payload_weight : float
        Payload weight in kg.

    battery_weight : float, optional
        Absolute battery weight in kg.

    battery_fraction : float, optional
        Battery weight as a fraction of gross weight.

    a, b : float
        Historical regression coefficients.

    initial_guess : float
        Initial gross-weight estimate in kg.
        Used to establish the root-finding bracket.

    tolerance : float
        Root-finding tolerance.

    max_iterations : int
        Maximum number of root-finding iterations.

    Returns
    -------
    EmptyWeightResult
        Estimated aircraft gross weight, empty weight, weight fractions,
        iteration count, and convergence status.
    """

    # ---------------------------------------------------------
    # 1. Validate inputs
    # ---------------------------------------------------------

    if (battery_weight is None) == (battery_fraction is None):
        raise ValueError(
            "Provide exactly one of battery_weight or battery_fraction."
        )

    if not np.isfinite(payload_weight) or payload_weight < 0:
        raise ValueError("Payload weight must be a finite, non-negative value.")

    if not np.isfinite(a) or not (0 < a < 1):
        raise ValueError(
            "Regression coefficient 'a' must be between 0 and 1 "
            "for this root-finding formulation."
        )

    if not np.isfinite(b):
        raise ValueError("Regression coefficient 'b' must be finite.")

    if not np.isfinite(initial_guess) or initial_guess <= 0:
        raise ValueError("Initial gross weight must be positive and finite.")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be positive and finite.")

    if max_iterations <= 0:
        raise ValueError("Maximum iterations must be positive.")

    # Determine which battery input method is being used.

    if battery_fraction is not None:

        if not np.isfinite(battery_fraction):
            raise ValueError("Battery fraction must be finite.")

        if not (0 <= battery_fraction < 1):
            raise ValueError("Battery fraction must be between 0 and 1.")

        fb = float(battery_fraction)

        # Battery weight varies with gross weight.
        # Wbattery = fb * W0

        def battery_weight_at(w0: float) -> float:
            return fb * w0

    else:

        if battery_weight is None:
            raise ValueError("Battery weight must be provided.")

        if not np.isfinite(battery_weight) or battery_weight < 0:
            raise ValueError(
                "Battery weight must be a finite, non-negative value."
            )

        wb = float(battery_weight)

        # Battery weight is fixed in kg.

        def battery_weight_at(w0: float) -> float:
            return wb

    # ---------------------------------------------------------
    # 2. Calculate empty weight from historical regression
    # ---------------------------------------------------------

    def calculate_empty_weight(w0: float) -> float:
        ln_we = a * np.log(w0) + b
        return float(np.exp(ln_we))

    # ---------------------------------------------------------
    # 3. Define weight-balance residual
    # ---------------------------------------------------------

    def weight_balance(w0: float) -> float:

        we = calculate_empty_weight(w0)
        wb = battery_weight_at(w0)

        # Weight balance:
        #
        # W0 = Wpayload + Wbattery + We
        #
        # Rearranged:
        #
        # f(W0) = W0 - Wpayload - Wbattery - We = 0

        return w0 - payload_weight - wb - we

    # ---------------------------------------------------------
    # 4. Establish a bracket for root finding
    # ---------------------------------------------------------

    lower = 1e-12
    upper = float(max(initial_guess, 1.0))

    # Increase upper bound until the residual becomes positive.

    while weight_balance(upper) <= 0:

        upper *= 2.0

        if upper > 1e15:
            raise ValueError(
                "Unable to find a valid gross-weight solution. "
                "Check the regression coefficients and weight inputs."
            )

    # ---------------------------------------------------------
    # 5. Solve for gross weight using Brent's method
    # ---------------------------------------------------------

    solution = root_scalar(
        weight_balance,
        bracket=(float(lower), float(upper)),
        method="brentq",
        xtol=float(tolerance),
        maxiter=int(max_iterations),
    )

    if not solution.converged:
        raise RuntimeError("Gross-weight estimation did not converge.")

    w0 = float(solution.root)

    # ---------------------------------------------------------
    # 6. Calculate final aircraft weights and fractions
    # ---------------------------------------------------------

    we = calculate_empty_weight(w0)
    wb = battery_weight_at(w0)

    empty_weight_fraction = we / w0
    payload_weight_fraction = payload_weight / w0
    battery_weight_fraction = wb / w0

    # ---------------------------------------------------------
    # 7. Return results
    # ---------------------------------------------------------

    return EmptyWeightResult(
        gross_weight=w0,
        empty_weight=we,
        empty_weight_fraction=empty_weight_fraction,
        payload_weight=payload_weight,
        payload_fraction=payload_weight_fraction,
        battery_weight=wb,
        battery_fraction=battery_weight_fraction,
        iterations=solution.iterations,
        converged=solution.converged,
    )


# =============================================================
# GRID VERSION
# =============================================================

def estimate_gross_weight_grid(
    payload_weight,
    battery_weight=None,
    battery_fraction=None,
    initial_guess=15.0,
    a=CONSTANT_A,
    b=CONSTANT_B,
    tolerance=1e-6,
    max_iterations=1000,
):
    """
    Perform empty-weight estimation over multiple design points.

    Inputs can be scalars or broadcast-compatible NumPy arrays.

    Exactly one of battery_weight or battery_fraction must be provided.
    """

    if (battery_weight is None) == (battery_fraction is None):
        raise ValueError(
            "Provide exactly one of battery_weight or battery_fraction."
        )

    payload_weight = np.asarray(payload_weight, dtype=float)
    initial_guess = np.asarray(initial_guess, dtype=float)

    if battery_fraction is not None:

        payload_weight, battery_fraction, initial_guess = (
            np.broadcast_arrays(
                payload_weight,
                np.asarray(battery_fraction, dtype=float),
                initial_guess,
            )
        )

        battery_input = battery_fraction

    else:

        payload_weight, battery_weight, initial_guess = (
            np.broadcast_arrays(
                payload_weight,
                np.asarray(battery_weight, dtype=float),
                initial_guess,
            )
        )

        battery_input = battery_weight

    shape = payload_weight.shape

    # Allocate output arrays.

    gross_weight = np.empty(shape)
    empty_weight = np.empty(shape)
    empty_weight_fraction = np.empty(shape)
    payload_weight_result = np.empty(shape)
    payload_fraction_result = np.empty(shape)
    battery_weight_result = np.empty(shape)
    battery_fraction_result = np.empty(shape)

    iterations = np.empty(shape, dtype=int)
    converged = np.empty(shape, dtype=bool)

    # ---------------------------------------------------------
    # Evaluate each design point
    # ---------------------------------------------------------

    for index in np.ndindex(shape):

        if battery_fraction is not None:

            result = estimate_gross_weight(
                payload_weight=float(payload_weight[index]),
                battery_fraction=float(battery_input[index]),
                a=a,
                b=b,
                initial_guess=float(initial_guess[index]),
                tolerance=tolerance,
                max_iterations=max_iterations,
            )

        else:

            result = estimate_gross_weight(
                payload_weight=float(payload_weight[index]),
                battery_weight=float(battery_input[index]),
                a=a,
                b=b,
                initial_guess=float(initial_guess[index]),
                tolerance=tolerance,
                max_iterations=max_iterations,
            )

        gross_weight[index] = result.gross_weight
        empty_weight[index] = result.empty_weight
        empty_weight_fraction[index] = result.empty_weight_fraction

        payload_weight_result[index] = result.payload_weight
        payload_fraction_result[index] = result.payload_fraction

        battery_weight_result[index] = result.battery_weight
        battery_fraction_result[index] = result.battery_fraction

        iterations[index] = result.iterations
        converged[index] = result.converged

    return {
        "gross_weight": gross_weight,
        "empty_weight": empty_weight,
        "empty_weight_fraction": empty_weight_fraction,
        "payload_weight": payload_weight_result,
        "payload_fraction": payload_fraction_result,
        "battery_weight": battery_weight_result,
        "battery_fraction": battery_fraction_result,
        "iterations": iterations,
        "converged": converged,
    }


# ---------------------------------------------------------
# Payload estimation based on gross weight
# ---------------------------------------------------------

@dataclass(frozen=True)
class PayloadCapacityResult:
    gross_weight: float
    empty_weight: float
    battery_weight: float
    payload_weight: float

    empty_weight_fraction: float
    battery_fraction: float
    payload_fraction: float


def estimate_payload_capacity(
    gross_weight: float,
    battery_weight: float,
    a: float = CONSTANT_A,
    b: float = CONSTANT_B,
) -> PayloadCapacityResult:

    """
    Reverse weight estimation.

    Given:
        Gross weight
        Fixed battery weight

    Calculate:
        Empty weight
        Available payload weight
        Corresponding weight fractions
    """

    if not np.isfinite(gross_weight) or gross_weight <= 0:
        raise ValueError("Gross weight must be positive.")

    if not np.isfinite(battery_weight) or battery_weight < 0:
        raise ValueError("Battery weight must be non-negative.")

    if not np.isfinite(a) or not (0 < a < 1):
        raise ValueError("Invalid regression coefficient a.")

    if not np.isfinite(b):
        raise ValueError("Invalid regression coefficient b.")

    # Empty weight regression
    empty_weight = np.exp(
        a * np.log(gross_weight) + b
    )

    # Remaining payload capacity
    payload_weight = (
        gross_weight
        - empty_weight
        - battery_weight
    )

    if payload_weight < 0:
        raise ValueError(
            "Empty weight and battery weight exceed gross weight."
        )

    return PayloadCapacityResult(
        gross_weight=gross_weight,
        empty_weight=empty_weight,
        battery_weight=battery_weight,
        payload_weight=payload_weight,

        empty_weight_fraction=empty_weight / gross_weight,
        battery_fraction=battery_weight / gross_weight,
        payload_fraction=payload_weight / gross_weight,
    )

# =============================================================
# TESTING
# =============================================================

if __name__ == "__main__":

    result = estimate_gross_weight(
        payload_weight=7.58,       # kg
        battery_weight=0.24,       # kg
        initial_guess=15.0,        # kg
    )

    print("\nEMPTY WEIGHT ESTIMATION")
    print("-----------------------")

    print(f"Gross Weight:          {result.gross_weight:.4f} kg")
    print(f"Empty Weight:          {result.empty_weight:.4f} kg")
    print(f"Empty Weight Fraction: {result.empty_weight_fraction:.4f}")

    print(f"Payload Weight:        {result.payload_weight:.4f} kg")
    print(f"Payload Fraction:      {result.payload_fraction:.4f}")

    print(f"Battery Weight:        {result.battery_weight:.4f} kg")
    print(f"Battery Fraction:      {result.battery_fraction:.4f}")

    print(f"Iterations:            {result.iterations}")
    print(f"Converged:             {result.converged}")