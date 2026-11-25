
import csv

import numpy as np

from utils.paths import OUTPUT_DIR

from analysis.constraint_analysis import (
    main as run_constraint_analysis,
    derived_values,
)

from analysis.weight_estimation import (
    estimate_gross_weight_grid,
)

from tools.scoring import optimize_bottles

from plotting.flight_score_plot import (
    plot_score_surfaces,
    plot_feasibility,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

# Battery
BATTERY_WEIGHT_KG = 0.24

# Bottle properties
FULL_BOTTLE_WEIGHT_KG = 1.81437
EMPTY_BOTTLE_WEIGHT_KG = 0.453592

FULL_BOTTLE_SCORE = 11
EMPTY_BOTTLE_SCORE = 3

# Maximum combined number of bottles.
# One bottle configuration is evaluated for every value from
# MAX_BOTTLES_START to MAX_BOTTLES_END (inclusive) in steps of 
# MAX_BOTTLES_STEP.
MAX_BOTTLES_START = 1
MAX_BOTTLES_END = 7
MAX_BOTTLES_STEP = 1

MAX_BOTTLES = np.arange(
    MAX_BOTTLES_START,
    MAX_BOTTLES_END + 1,
    MAX_BOTTLES_STEP,
    dtype=int
)

# Output
SCORING_OUTPUT_FILE = OUTPUT_DIR / "scoring_summary.csv"
PLOT_OUTPUT_DIR = OUTPUT_DIR / "plots"


# ==========================================================
# MAIN
# ==========================================================

def main():

    if MAX_BOTTLES.size == 0:
        raise ValueError(
            "MAX_BOTTLES is empty. Check MAX_BOTTLES_START, "
            "MAX_BOTTLES_END, and MAX_BOTTLES_STEP."
        )

    n_config = MAX_BOTTLES.size

    # ------------------------------------------------------
    # 1. Run constraint analysis
    # ------------------------------------------------------

    results = run_constraint_analysis()

    p = results["parameters"]

    AR = results["AR"]
    V = results["V"]

    design_points = results["design_points"]

    tw = results["tw"]

    # ------------------------------------------------------
    # 2. Create AR-V design grids
    # ------------------------------------------------------

    AR_grid, V_grid = np.meshgrid(
        AR[:, 0],
        V[0, :],
        indexing="ij",
    )

    n_ar, n_v = AR_grid.shape

    # ------------------------------------------------------
    # 3. Determine design feasibility
    # ------------------------------------------------------

    feasible_grid = np.zeros(
        (n_ar, n_v),
        dtype=bool,
    )

    for i_ar in range(n_ar):

        for i_v in range(n_v):

            feasible_grid[i_ar, i_v] = (
                design_points[(i_ar, i_v)] is not None
            )

    # ------------------------------------------------------
    # 4. Estimate gross weight for every AR-V combination
    # ------------------------------------------------------

    payload_weight_grid = np.full(
        (n_ar, n_v),
        float(p["payload_mass_historical_kg"]),
    )

    weight_results = estimate_gross_weight_grid(
        payload_weight=payload_weight_grid,
        battery_weight=BATTERY_WEIGHT_KG,
        initial_guess=float(p["mtow_mass_kg"]),
    )

    gross_weight_grid = weight_results["gross_weight"]
    empty_weight_grid = weight_results["empty_weight"]
    battery_weight_grid = weight_results["battery_weight"]
    payload_fraction_grid = weight_results["payload_fraction"]
    empty_weight_fraction_grid = weight_results["empty_weight_fraction"]
    battery_fraction_grid = weight_results["battery_fraction"]

    print("\n" + "=" * 60)
    print("WEIGHT ESTIMATION")
    print("=" * 60)

    print(
        f"Gross weight:          "
        f"{gross_weight_grid[0, 0]:.4f} kg"
    )

    print(
        f"Empty weight:          "
        f"{empty_weight_grid[0, 0]:.4f} kg"
    )

    print(
        f"Battery weight:        "
        f"{battery_weight_grid[0, 0]:.4f} kg"
    )

    print(
        f"Payload weight:        "
        f"{weight_results['payload_weight'][0, 0]:.4f} kg"
    )

    print(
        f"Payload fraction:      "
        f"{payload_fraction_grid[0, 0]:.4f}"
    )

    print(
        f"Battery fraction:      "
        f"{battery_fraction_grid[0, 0]:.4f}"
    )

    # ------------------------------------------------------
    # 5. Initialize output arrays
    # ------------------------------------------------------

    # Bottle result have one layer per MAX_BOTTLES configuration:
    # shape = (n_config, n_ar, n_v)

    score_grid = np.full(
        (n_config, n_ar, n_v),
        np.nan,
    )

    cruise_power_grid = np.full(
        (n_ar, n_v),
        np.nan,
    )

    cruise_drag_grid = np.full(
        (n_ar, n_v),
        np.nan,
    )

    wing_area_grid = np.full(
        (n_ar, n_v),
        np.nan,
    )

    span_grid = np.full(
        (n_ar, n_v),
        np.nan,
    )

    full_bottle_grid = np.full(
        (n_config, n_ar, n_v),
        np.nan,
    )

    empty_bottle_grid = np.full(
        (n_config, n_ar, n_v),
        np.nan,
    )

    # ------------------------------------------------------
    # 6. Evaluate every feasible AR-V combination
    # ------------------------------------------------------

    summary = []

    for i_ar in range(n_ar):

        for i_v in range(n_v):

            ar = float(AR_grid[i_ar, i_v])

            velocity = float(V_grid[i_ar, i_v])

            dp = design_points[(i_ar, i_v)]

            weight_summary = {

                "gross_weight_kg": gross_weight_grid[i_ar, i_v],
                "empty_weight_kg": empty_weight_grid[i_ar, i_v],
                "battery_weight_kg": battery_weight_grid[i_ar, i_v],
                "payload_weight_kg": weight_results["payload_weight"][i_ar, i_v],
                "empty_weight_fraction": empty_weight_fraction_grid[i_ar, i_v],
                "battery_fraction": battery_fraction_grid[i_ar, i_v],
                "payload_fraction": payload_fraction_grid[i_ar, i_v],

            }

            # ----------------------------------------------
            # Infeasible design combination
            # ----------------------------------------------

            if dp is None:

                for max_bottles in MAX_BOTTLES:

                    summary.append({

                        "AR": ar,
                        "V_m_s": velocity,

                        "feasible": False,

                        "WS_design_N_m2": np.nan,
                        "PW_design_W_kg": np.nan,

                        "cruise_drag_N": np.nan,
                        "cruise_power_W": np.nan,

                        "wing_area_m2": np.nan,
                        "span_m": np.nan,

                        **weight_summary,

                        "max_bottles": int(max_bottles),

                        "full_bottles": np.nan,
                        "empty_bottles": np.nan,
                        "total_bottles": np.nan,

                        "flight_score": np.nan,

                        "governing": "",
                        "limited_by": "",

                    })

                continue

            # ----------------------------------------------
            # Extract T/W curves for this AR-V combination
            # ----------------------------------------------

            tw_2D = {

                name: curve[i_ar, i_v, :]

                for name, curve in tw.items()

            }

            # ----------------------------------------------
            # Determine required T/W at selected W/S
            # ----------------------------------------------

            tw_req = float(
                np.fmax.reduce(
                    np.vstack(
                        list(tw_2D.values())
                    ),
                    axis=0,
                )[dp.index]
            )

            # ----------------------------------------------
            # Calculate aerodynamic and geometric values
            # ----------------------------------------------

            design_parameters = p.copy()
            design_parameters["mass_kg"] = float(
                gross_weight_grid[i_ar, i_v]
            )

            derived = derived_values(
                p=design_parameters,
                dp=dp,
                tw_req=tw_req,
                ar=ar,
                velocity=velocity,
            )

            geometry = derived["WING GEOMETRY"]

            propulsion = derived["PROPULSION"]

            cruise_drag = float(
                propulsion["cruise_drag_N"]
            )

            cruise_power = float(
                propulsion["cruise_power_W"]
            )

            wing_area = float(
                geometry["wing_area_m2"]
            )

            span = float(
                geometry["span_m"]
            )

            cruise_power_grid[i_ar, i_v] = cruise_power

            cruise_drag_grid[i_ar, i_v] = cruise_drag

            wing_area_grid[i_ar, i_v] = wing_area

            span_grid[i_ar, i_v] = span

            # ----------------------------------------------
            # Optimize bottle configuration for every
            # maximum bottle count
            # ----------------------------------------------

            for i_cfg, max_bottles in enumerate(MAX_BOTTLES):

                bottle_result = optimize_bottles(

                    payload_weight=float(
                        weight_results["payload_weight"][i_ar, i_v]
                    ),

                    full_bottle_weight=FULL_BOTTLE_WEIGHT_KG,

                    empty_bottle_weight=EMPTY_BOTTLE_WEIGHT_KG,

                    full_bottle_score=FULL_BOTTLE_SCORE,

                    empty_bottle_score=EMPTY_BOTTLE_SCORE,

                    max_bottles=int(max_bottles),

                )

                # ----------------------------------------------
                # Store results in grids
                # ----------------------------------------------

                score_grid[i_cfg, i_ar, i_v] = (
                    bottle_result.flight_score
                )

                full_bottle_grid[i_cfg, i_ar, i_v] = (
                    bottle_result.full_bottles
                )

                empty_bottle_grid[i_cfg, i_ar, i_v] = (
                    bottle_result.empty_bottles
                )

                # ----------------------------------------------
                # Store summary information
                # ----------------------------------------------

                summary.append({

                    "AR": ar,
                    "V_m_s": velocity,

                    "feasible": True,

                    "WS_design_N_m2": dp.ws,
                    "PW_design_W_kg": dp.pw,

                    "cruise_drag_N": cruise_drag,
                    "cruise_power_W": cruise_power,

                    "wing_area_m2": wing_area,
                    "span_m": span,

                    **weight_summary,

                    "max_bottles": int(max_bottles),

                    "full_bottles": bottle_result.full_bottles,
                    "empty_bottles": bottle_result.empty_bottles,

                    "total_bottles": (
                        bottle_result.full_bottles
                        + bottle_result.empty_bottles
                    ),

                    "flight_score": bottle_result.flight_score,

                    "governing": dp.governing,
                    "limited_by": dp.limited_by,

                })

    # ------------------------------------------------------
    # 7. Export scoring summary CSV
    # ------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        SCORING_OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys(),
        )

        writer.writeheader()

        writer.writerows(summary)

    # ------------------------------------------------------
    # 8. Generate plots
    # ------------------------------------------------------

    PLOT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Flight score surfaces (one per bottle configuration)
    plot_score_surfaces(

        AR_grid=AR_grid,

        V_grid=V_grid,

        score_grids=score_grid,

        feasible_grid=feasible_grid,

        labels=[
            f"max {max_bottles} bottle{'s' if max_bottles != 1 else ''}"
            for max_bottles in MAX_BOTTLES
        ],

        save_path=(
            PLOT_OUTPUT_DIR / "flight_score_surface.png"
        ),

        show=True,

    )

    # Design feasibility
    plot_feasibility(

        AR_grid=AR_grid,

        V_grid=V_grid,

        feasible_grid=feasible_grid,

        save_path=(
            PLOT_OUTPUT_DIR / "design_feasibility.png"
        ),

        show=True,

    )

    # ------------------------------------------------------
    # 9. Identify best design point for each configuration
    # ------------------------------------------------------

    if np.any(np.isfinite(score_grid)):

        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)

        print(
            f"{'Max':>5} {'Score':>7} {'Full':>6} {'Empty':>6} "
            f"{'Total':>6} {'AR':>8} {'V [m/s]':>9}"
        )

        for i_cfg, max_bottles in enumerate(MAX_BOTTLES):

            best_index = np.unravel_index(
                np.nanargmax(score_grid[i_cfg]),
                score_grid[i_cfg].shape,
            )

            best_full = full_bottle_grid[i_cfg][best_index]

            best_empty = empty_bottle_grid[i_cfg][best_index]

            print(
                f"{max_bottles:>5d} "
                f"{score_grid[i_cfg][best_index]:>7.0f} "
                f"{best_full:>6.0f} "
                f"{best_empty:>6.0f} "
                f"{best_full + best_empty:>6.0f} "
                f"{AR_grid[best_index]:>8.4f} "
                f"{V_grid[best_index]:>9.4f}"
            )

        # Overall best across all configurations

        best_cfg, best_ar, best_v = np.unravel_index(
            np.nanargmax(score_grid),
            score_grid.shape,
        )

        print("-" * 60)

        print(f"Best max bottles:        {MAX_BOTTLES[best_cfg]}")

        print(f"Best aspect ratio:       {AR_grid[best_ar, best_v]:.4f}")

        print(
            f"Best cruise velocity:    "
            f"{V_grid[best_ar, best_v]:.4f} m/s"
        )

        print(
            f"Maximum flight score:    "
            f"{score_grid[best_cfg, best_ar, best_v]:.0f}"
        )

    else:

        print("\nNo feasible design points were found.")

    # ------------------------------------------------------
    # 10. Completion summary
    # ------------------------------------------------------

    feasible_count = np.count_nonzero(feasible_grid)

    total_count = feasible_grid.size

    print("\n" + "=" * 60)
    print("SCORING ANALYSIS COMPLETED")
    print("=" * 60)

    print(f"AR points:             {n_ar}")

    print(f"Velocity points:       {n_v}")

    print(
        f"Feasible combinations: "
        f"{feasible_count}/{total_count}"
    )

    print(
        f"Payload weight:        "
        f"{weight_results['payload_weight'][0, 0]:.4f} kg"
    )

    print(f"Maximum bottles:       {MAX_BOTTLES.tolist()}")

    print(f"CSV output:            {SCORING_OUTPUT_FILE}")

    print(f"Plot output:           {PLOT_OUTPUT_DIR}")

    print("=" * 60)


# ==========================================================
# SCRIPT EXECUTION
# ==========================================================

if __name__ == "__main__":
    main()
