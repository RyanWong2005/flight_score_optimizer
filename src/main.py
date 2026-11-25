
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

from limitations.geometry import wingspan_is_feasible
from limitations.performance import (
    cruise_stall_speed_is_feasible,
    gross_weight_is_feasible,
)

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

# Total bottle slots carried in every configuration. Each configuration
# varies the number of full bottles; all remaining slots carry empty bottles.
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

    if (
        MAX_BOTTLES_END < 1
        or MAX_BOTTLES_START < 1
        or MAX_BOTTLES_STEP < 1
        or MAX_BOTTLES.size == 0
        or MAX_BOTTLES[-1] > MAX_BOTTLES_END
    ):
        raise ValueError(
            "Bottle configurations must contain between 1 and "
            "MAX_BOTTLES_END full bottles, with a positive step."
        )

    n_config = MAX_BOTTLES.size
    empty_bottle_counts = MAX_BOTTLES_END - MAX_BOTTLES
    payload_weight_by_config = (
        MAX_BOTTLES * FULL_BOTTLE_WEIGHT_KG
        + empty_bottle_counts * EMPTY_BOTTLE_WEIGHT_KG
    )

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

    design_feasible_grid = np.zeros((n_ar, n_v), dtype=bool)

    for i_ar in range(n_ar):

        for i_v in range(n_v):

            design_feasible_grid[i_ar, i_v] = (
                design_points[(i_ar, i_v)] is not None
            )

    # ------------------------------------------------------
    # 4. Estimate gross weight for every AR-V combination
    # ------------------------------------------------------

    payload_weight_grid = np.broadcast_to(
        payload_weight_by_config[:, None, None],
        (n_config, n_ar, n_v),
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

    for i_cfg, full_bottle_count in enumerate(MAX_BOTTLES):
        print(
            f"{full_bottle_count} full + "
            f"{empty_bottle_counts[i_cfg]} empty: "
            f"payload {payload_weight_by_config[i_cfg]:.6f} kg, "
            f"gross {gross_weight_grid[i_cfg, 0, 0]:.4f} kg"
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

    feasible_grid = np.broadcast_to(
        design_feasible_grid,
        (n_config, n_ar, n_v),
    ).copy()

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

            # ----------------------------------------------
            # Infeasible design combination
            # ----------------------------------------------

            if dp is None:

                for i_cfg, full_bottle_count in enumerate(MAX_BOTTLES):

                    summary.append({

                        "AR": ar,
                        "V_m_s": velocity,

                        "feasible": False,

                        "WS_design_N_m2": np.nan,
                        "PW_design_W_kg": np.nan,

                        "cruise_drag_N": np.nan,
                        "cruise_power_W": np.nan,

                        "stall_speed_m_s": np.nan,
                        "wing_area_m2": np.nan,
                        "span_m": np.nan,

                        "gross_weight_kg": gross_weight_grid[i_cfg, i_ar, i_v],
                        "empty_weight_kg": empty_weight_grid[i_cfg, i_ar, i_v],
                        "battery_weight_kg": battery_weight_grid[i_cfg, i_ar, i_v],
                        "payload_weight_kg": weight_results["payload_weight"][i_cfg, i_ar, i_v],
                        "empty_weight_fraction": empty_weight_fraction_grid[i_cfg, i_ar, i_v],
                        "battery_fraction": battery_fraction_grid[i_cfg, i_ar, i_v],
                        "payload_fraction": payload_fraction_grid[i_cfg, i_ar, i_v],

                        "full_bottle_config": int(full_bottle_count),

                        "full_bottles": int(full_bottle_count),
                        "empty_bottles": int(empty_bottle_counts[i_cfg]),
                        "total_bottles": MAX_BOTTLES_END,

                        "flight_score": np.nan,

                        "governing": "",
                        "limited_by": "",
                        "limitations_failed": "",

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

            # Every configuration occupies all bottle slots. The configured
            # number are full and all remaining slots carry empty bottles.
            for i_cfg, full_bottle_count in enumerate(MAX_BOTTLES):

                gross_weight = float(gross_weight_grid[i_cfg, i_ar, i_v])
                design_parameters = p.copy()
                design_parameters["mass_kg"] = gross_weight

                derived = derived_values(
                    p=design_parameters,
                    dp=dp,
                    tw_req=tw_req,
                    ar=ar,
                    velocity=velocity,
                )

                geometry = derived["WING GEOMETRY"]
                aerodynamics = derived["AERODYNAMICS"]
                propulsion = derived["PROPULSION"]

                cruise_drag = float(propulsion["cruise_drag_N"])
                cruise_power = float(propulsion["cruise_power_W"])
                wing_area = float(geometry["wing_area_m2"])
                span = float(geometry["span_m"])
                stall_speed = float(aerodynamics["V_stall_m_s"])

                failed_limitations = []

                if not wingspan_is_feasible(span):
                    failed_limitations.append("wingspan >= 96 in")

                if not cruise_stall_speed_is_feasible(
                    velocity,
                    stall_speed,
                ):
                    failed_limitations.append(
                        "cruise speed < 1.3 x stall speed"
                    )

                if not gross_weight_is_feasible(gross_weight):
                    failed_limitations.append(
                        "gross weight >= 24.94331066 kg"
                    )

                is_feasible = not failed_limitations
                feasible_grid[i_cfg, i_ar, i_v] = is_feasible

                empty_bottle_count = int(empty_bottle_counts[i_cfg])
                full_bottle_grid[i_cfg, i_ar, i_v] = full_bottle_count
                empty_bottle_grid[i_cfg, i_ar, i_v] = empty_bottle_count

                flight_score = (
                    int(full_bottle_count) * FULL_BOTTLE_SCORE
                    + empty_bottle_count * EMPTY_BOTTLE_SCORE
                )

                if is_feasible:
                    score_grid[i_cfg, i_ar, i_v] = flight_score

                weight_summary = {
                    "gross_weight_kg": gross_weight,
                    "empty_weight_kg": empty_weight_grid[i_cfg, i_ar, i_v],
                    "battery_weight_kg": battery_weight_grid[i_cfg, i_ar, i_v],
                    "payload_weight_kg": weight_results["payload_weight"][i_cfg, i_ar, i_v],
                    "empty_weight_fraction": empty_weight_fraction_grid[i_cfg, i_ar, i_v],
                    "battery_fraction": battery_fraction_grid[i_cfg, i_ar, i_v],
                    "payload_fraction": payload_fraction_grid[i_cfg, i_ar, i_v],
                }

                summary.append({
                    "AR": ar,
                    "V_m_s": velocity,
                    "feasible": is_feasible,
                    "WS_design_N_m2": dp.ws,
                    "PW_design_W_kg": dp.pw,
                    "cruise_drag_N": cruise_drag,
                    "cruise_power_W": cruise_power,
                    "stall_speed_m_s": stall_speed,
                    "wing_area_m2": wing_area,
                    "span_m": span,
                    **weight_summary,
                    "full_bottle_config": int(full_bottle_count),
                    "full_bottles": int(full_bottle_count),
                    "empty_bottles": empty_bottle_count,
                    "total_bottles": MAX_BOTTLES_END,
                    "flight_score": flight_score if is_feasible else np.nan,
                    "governing": dp.governing,
                    "limited_by": dp.limited_by,
                    "limitations_failed": "; ".join(failed_limitations),
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

        feasible_grid=design_feasible_grid,

        labels=[
            f"{full_bottles} full + "
            f"{MAX_BOTTLES_END - full_bottles} empty"
            for full_bottles in MAX_BOTTLES
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

        feasible_grid=np.any(feasible_grid, axis=0),

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
            f"{'Full cfg':>8} {'Score':>7} {'Full':>6} {'Empty':>6} "
            f"{'Total':>6} {'AR':>8} {'V [m/s]':>9}"
        )

        for i_cfg, full_bottle_count in enumerate(MAX_BOTTLES):

            if not np.any(np.isfinite(score_grid[i_cfg])):
                print(
                    f"{full_bottle_count:>5d} "
                    f"{'n/a':>7} {'n/a':>6} {'n/a':>6} {'n/a':>6}"
                )
                continue

            best_index = np.unravel_index(
                np.nanargmax(score_grid[i_cfg]),
                score_grid[i_cfg].shape,
            )

            best_full = full_bottle_grid[i_cfg][best_index]

            best_empty = empty_bottle_grid[i_cfg][best_index]

            print(
                f"{full_bottle_count:>5d} "
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

        print(
            f"Best full-bottle config: "
            f"{MAX_BOTTLES[best_cfg]} full + "
            f"{empty_bottle_counts[best_cfg]} empty"
        )

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

    feasible_count = np.count_nonzero(np.any(feasible_grid, axis=0))

    total_count = design_feasible_grid.size

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
        f"Payload sweep:         "
        f"{payload_weight_by_config.min():.4f}–"
        f"{payload_weight_by_config.max():.4f} kg"
    )

    print(
        f"Full-bottle configs:   "
        f"{MAX_BOTTLES.tolist()} (each carries {MAX_BOTTLES_END} bottles)"
    )

    print(f"CSV output:            {SCORING_OUTPUT_FILE}")

    print(f"Plot output:           {PLOT_OUTPUT_DIR}")

    print("=" * 60)


# ==========================================================
# SCRIPT EXECUTION
# ==========================================================

if __name__ == "__main__":
    main()
