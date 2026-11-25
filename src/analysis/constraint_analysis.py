
import csv
from pathlib import Path

import numpy as np
import numpy.typing as npt

from utils.paths import INPUT_DATA, OUTPUT_DIR

from utils.database_reader import load_database

from tools.grid import build_grid
from tools.design_point import DesignPoint, design_point

from tools.TW_PW_conversion import (
    tw_to_pw,
    speed_at_cl,
    pw_per_n_to_per_kg,
)

from aerodynamics.aero import (
    lift_induced_drag_factor, 
    dynamic_pressure
    ,dynamic_pressure_from_velocity
)
from plotting.PW_WS_plot import plot_baseline

from constraints.cruise_airspeed import cruise_speed_tw
from constraints.clmax_stall import stall_ws_max_clean
from constraints.ground_run_distance import ground_run_dist_tw
from constraints.level_const_velocity_turn import level_turn_tw
from constraints.max_angle_of_climb import max_aoc_tw
from constraints.stall_speed import stall_ws_max
from constraints.ldg_distance import ldg_dist_ws


Array = npt.NDArray[np.float64]

SEP = "=" * 60


# ==========================================================
# CONFIGURATION
# ==========================================================

# Design point selection strategy:
# "max_ws" = maximum feasible wing loading
# "min_pw" = minimum power loading requirement
STRATEGY = "max_ws"

# Wing loading design space
WS_VALUES = np.linspace(2.0, 250.0, 250)

# Plot selection
PLOT_AR_INDEX = 10
PLOT_V_INDEX = 10


# ==========================================================
# CONSTRAINT ANALYSIS
# ==========================================================

def run_case(p, AR, V, WS):
    """
    Calculate T/W and P/W constraint curves over the
    AR, V and W/S design space.

    Expected broadcast dimensions:
        AR: (n_AR, 1, 1)
        V:  (1, n_V, 1)
        WS: (1, 1, n_WS)

    Resulting constraint arrays should broadcast to:
        (n_AR, n_V, n_WS)
    """

    rho = p["competition_density_kg_m3"]
    gamma = np.radians(p["climb_angle_degree"])
    cd0 = p["cd_min_unitless"]
    eta = p["eta_total_unitless"]

    # ------------------------------------------------------
    # Aerodynamic calculations
    # ------------------------------------------------------

    k = lift_induced_drag_factor(AR, p)

    cl_star = np.sqrt(cd0 / k)

    # ------------------------------------------------------
    # Dynamic pressure from cruise velocity
    # ------------------------------------------------------

    q_cruise = dynamic_pressure_from_velocity(
        rho,
        V,
    )

    # ------------------------------------------------------
    # Flight velocities
    # ------------------------------------------------------

    v_cruise = V

    v_lof = 1.1 * speed_at_cl(
        WS,
        p["cl_max_TO_unitless"],
        rho,
    )

    v_aoc = speed_at_cl(
        WS * np.cos(gamma),
        cl_star,
        rho,
    )

    # ------------------------------------------------------
    # Thrust-to-weight constraints
    # ------------------------------------------------------

    tw = {

        "level_turn": level_turn_tw(
            WS, AR, k, q_cruise, p
        ),

        "cruise_airspeed": cruise_speed_tw(
            WS, AR, k, q_cruise, p
        ),

        "max_aoc": max_aoc_tw(
            WS, AR, k, p
        ),

        "ground_run_distance": ground_run_dist_tw(
            WS, p
        ),

    }

    # ------------------------------------------------------
    # Velocities used for P/W conversion
    # ------------------------------------------------------

    velocities = {

        "level_turn": v_cruise,

        "cruise_airspeed": v_cruise,

        "max_aoc": v_aoc,

        "ground_run_distance": 0.7 * v_lof,

    }

    # ------------------------------------------------------
    # Convert T/W to P/W
    # ------------------------------------------------------

    pw = {

        name: pw_per_n_to_per_kg(
            tw_to_pw(
                curve,
                velocities[name],
                eta,
            )
        )

        for name, curve in tw.items()

    }

    # ------------------------------------------------------
    # Wing loading limits
    # ------------------------------------------------------

    ws_limits = {

        "stall": stall_ws_max(p),

        "ldg_distance": ldg_dist_ws(p),

    }

    return tw, pw, ws_limits, q_cruise


# ==========================================================
# DESIGN POINT SELECTION
# ==========================================================

def select_design_points(
    ws,
    pw_curves,
    ws_limits,
    strategy="max_ws",
):
    """
    Select one design point for every AR/V combination.
    """

    n_ar, n_v, _ = next(
        iter(pw_curves.values())
    ).shape

    design_points = {}

    ws_selected = np.full(
        (n_ar, n_v),
        np.nan,
    )

    pw_selected = np.full(
        (n_ar, n_v),
        np.nan,
    )

    for i_ar in range(n_ar):

        for i_v in range(n_v):

            # Extract P/W curves for this AR/V combination
            pw_curves_2D = {

                name: curve[i_ar, i_v, :]

                for name, curve in pw_curves.items()

            }

            try:

                dp = design_point(
                    ws=ws,
                    pw_curves=pw_curves_2D,
                    ws_limits=ws_limits,
                    strategy=strategy,
                )

            except ValueError as exc:

                # Infeasible design combinations are retained
                # in the design-space results as None.
                if str(exc).startswith(
                    ("Infeasible:", "No feasible W/S")
                ):

                    design_points[(i_ar, i_v)] = None

                    continue

                raise

            design_points[(i_ar, i_v)] = dp

            ws_selected[i_ar, i_v] = dp.ws

            pw_selected[i_ar, i_v] = dp.pw

    return design_points, ws_selected, pw_selected


# ==========================================================
# DERIVED DESIGN VALUES
# ==========================================================

def derived_values(
    p,
    dp: DesignPoint,
    tw_req: float,
    ar: float,
    velocity: float,
):
    """
    Calculate aerodynamic, geometric and propulsion
    parameters for the selected design point.
    """

    g = p["gravity_sea_level_m_per_s2"]
    rho = p["competition_density_kg_m3"]
    mass = (
        p["mass_kg"]
        if "mass_kg" in p
        else p["mtow_mass_kg"]
    )

    cd0 = p["cd_min_unitless"]

    # Induced drag factor for selected aspect ratio
    k = float(
        lift_induced_drag_factor(
            np.asarray(ar),
            p,
        )
    )

    # Dynamic pressure at selected cruise velocity
    q_cr = 0.5 * rho * velocity**2

    # Aircraft weight
    weight = mass * g

    # Wing geometry
    wing_area = weight / dp.ws

    span = np.sqrt(ar * wing_area)

    # Cruise aerodynamics
    cl_cr = dp.ws / q_cr

    cd_cr = cd0 + k * cl_cr**2

    drag = q_cr * wing_area * cd_cr

    return {

        "AERODYNAMICS": {

            "CL_cruise": cl_cr,

            "CD_cruise": cd_cr,

            "LD_cruise": cl_cr / cd_cr,

            "CL_best_LD": np.sqrt(cd0 / k),

            "V_stall_m_s": np.sqrt(
                2 * dp.ws /
                (rho * p["cl_max_LDG_unitless"])
            ),

        },

        "WING GEOMETRY": {

            "weight_N": weight,

            "wing_area_m2": wing_area,

            "span_m": span,

        },

        "PROPULSION": {

            "power_required_W": dp.pw * mass,

            "peak_thrust_required_N": tw_req * weight,

            "cruise_drag_N": drag,

            "cruise_power_W": (
                drag * velocity /
                p["eta_total_unitless"]
            ),

        },

    }


# ==========================================================
# OUTPUT FILE GENERATION
# ==========================================================

def write_case_output(
    path: Path,
    p,
    units,
    ws,
    tw,
    pw,
    dp: DesignPoint,
    ar: float,
    velocity: float,
):
    """
    Write the constraint analysis results for one AR/V
    combination.
    """

    # Maximum T/W requirement at selected design W/S
    tw_req = float(
        np.fmax.reduce(
            np.vstack(list(tw.values())),
            axis=0,
        )[dp.index]
    )

    # Update input parameters for this design combination
    p_output = p.copy()

    p_output["aspect_ratio_unitless"] = ar

    p_output["target_cruise_speed_m_per_s"] = velocity

    p_output["dynamic_pressure_cruise_pa"] = (
        0.5 *
        p["competition_density_kg_m3"] *
        velocity**2
    )

    lines = [

        f"# {path.stem}",

        SEP,

        "DESIGN VARIABLES",

        f"aspect_ratio_unitless = {ar:.4g}",

        f"cruise_velocity_m_s = {velocity:.4g}",

        SEP,

        "INPUTS",

    ]

    # Write database parameters and corresponding units
    for name, val in p_output.items():

        unit = units.get(name, "")

        lines.append(
            f"{name} = {val} # {unit}"
        )

    # Constraint results
    lines += [

        SEP,

        "CONSTRAINT RESULT",

        f"WS_design_N_m2 = {dp.ws:.4g}",

        f"PW_design_W_kg = {dp.pw:.4g}",

        f"TW_peak_at_design_WS = {tw_req:.4g}",

        f"governing_constraint = {dp.governing}",

        f"WS_limit_N_m2 = {dp.ws_limit:.4g}",

        f"limited_by = {dp.limited_by}",

        f"at_limit = {dp.at_limit}",

    ]

    # P/W curves
    lines += [

        SEP,

        "P/W CURVES [W/kg]",

        "WS," + ",".join(pw),

    ]

    for i, w in enumerate(ws):

        lines.append(

            f"{w:.1f},"
            + ",".join(
                f"{pw[name][i]:.4g}"
                for name in pw
            )

        )

    # Derived values
    derived = derived_values(
        p_output,
        dp,
        tw_req,
        ar,
        velocity,
    )

    for section, values in derived.items():

        lines += [

            SEP,

            section,

        ]

        lines += [

            f"{name} = {val:.4g}"

            for name, val in values.items()

        ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ==========================================================
# MAIN
# ==========================================================

def main(*, close_plots=True):

    # ------------------------------------------------------
    # 1. Load baseline database
    # ------------------------------------------------------

    p, units = load_database(INPUT_DATA)

    p = dynamic_pressure(p)

    print(f"Loaded database: {INPUT_DATA.name}")

    # ------------------------------------------------------
    # 2. Build design space
    # ------------------------------------------------------

    AR, V = build_grid()

    AR_3D = AR[:, :, None]

    V_3D = V[:, :, None]

    WS_3D = WS_VALUES[None, None, :]

    clean_stall_ws_sweep = stall_ws_max_clean(AR, p)

    # ------------------------------------------------------
    # 3. Run constraint analysis
    # ------------------------------------------------------

    tw, pw_curves, ws_limits, q_cruise_grid = run_case(
        p,
        AR_3D,
        V_3D,
        WS_3D,
    )

    cl_max_required_at_stall_limit = (
        ws_limits["stall"] / p["dynamic_pressure_stall_pa"]
    )

    # Expected dimensions:
    # (number of AR values, number of V values, number of W/S values)

    shape = (
        AR.shape[0],
        V.shape[1],
        WS_VALUES.size,
    )

    # Broadcast constraint arrays to common dimensions
    tw = {

        name: np.broadcast_to(curve, shape)

        for name, curve in tw.items()

    }

    pw_curves = {

        name: np.broadcast_to(curve, shape)

        for name, curve in pw_curves.items()

    }

    # ------------------------------------------------------
    # 4. Select design points
    # ------------------------------------------------------

    design_points, ws_selected, pw_selected = (
        select_design_points(
            WS_VALUES,
            pw_curves,
            ws_limits,
            strategy=STRATEGY,
        )
    )

    # ------------------------------------------------------
    # 5. Create output directories
    # ------------------------------------------------------

    run_output_dir = OUTPUT_DIR / "runs"

    plot_output_dir = OUTPUT_DIR / "plots"

    run_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # 6. Write individual output files
    # ------------------------------------------------------

    summary = []

    for i_ar in range(AR.shape[0]):

        for i_v in range(V.shape[1]):

            ar = float(AR[i_ar, 0])

            velocity = float(V[0, i_v])

            dp = design_points[(i_ar, i_v)]

            # ----------------------------------------------
            # Infeasible design combination
            # ----------------------------------------------

            if dp is None:

                summary.append({

                    "AR": ar,

                    "V": velocity,

                    "dynamic_pressure_cruise_pa": float(
                        q_cruise_grid[0, i_v, 0]
                    ),

                    "feasible": False,

                    "WS_design": np.nan,

                    "PW_design": np.nan,

                    "governing": "",

                    "WS_limit": np.nan,

                    "limited_by": "",

                })

                continue

            # ----------------------------------------------
            # Extract corresponding curves
            # ----------------------------------------------

            tw_2D = {

                name: curve[i_ar, i_v, :]

                for name, curve in tw.items()

            }

            pw_2D = {

                name: curve[i_ar, i_v, :]

                for name, curve in pw_curves.items()

            }

            # ----------------------------------------------
            # Generate filename
            # ----------------------------------------------

            case_name = (

                f"AR_{ar:.2f}_V_{velocity:.2f}"

                .replace(".", "p")

            )

            # ----------------------------------------------
            # Write individual output
            # ----------------------------------------------

            write_case_output(

                path=run_output_dir / f"{case_name}.output",

                p=p,

                units=units,

                ws=WS_VALUES,

                tw=tw_2D,

                pw=pw_2D,

                dp=dp,

                ar=ar,

                velocity=velocity,

            )

            # ----------------------------------------------
            # Store summary information
            # ----------------------------------------------

            summary.append({

                "AR": ar,

                "V": velocity,

                "dynamic_pressure_cruise_pa": float(
                    q_cruise_grid[0, i_v, 0]
                ),

                "feasible": True,

                "WS_design": dp.ws,

                "PW_design": dp.pw,

                "governing": dp.governing,

                "WS_limit": dp.ws_limit,

                "limited_by": dp.limited_by,

            })

    # ------------------------------------------------------
    # 7. Write summary CSV
    # ------------------------------------------------------

    summary_path = OUTPUT_DIR / "design_space_summary.csv"

    with open(
        summary_path,
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
    # 8. Generate representative P/W plot
    # ------------------------------------------------------

    i_ar = PLOT_AR_INDEX

    i_v = PLOT_V_INDEX

    dp = design_points[(i_ar, i_v)]

    if dp is not None:

        pw_2D = {

            name: curve[i_ar, i_v, :]

            for name, curve in pw_curves.items()

        }

        plot_baseline(

            ws=WS_VALUES,

            pw_curves=pw_2D,

            ws_limits=ws_limits,

            dp=dp,

            path=plot_output_dir / "baseline.png",

            ar=float(AR[i_ar, 0]),

            velocity=float(V[0, i_v]),

            cl_max_required=(
                WS_VALUES / p["dynamic_pressure_stall_pa"]
            ),

            cl_max_available=(
                float(clean_stall_ws_sweep[i_ar, 0])
                / p["dynamic_pressure_stall_pa"]
            ),

            cl_max_required_at_stall_limit=(
                cl_max_required_at_stall_limit
            ),

            target_stall_speed=p["target_stall_speed_m_per_s"],

            close=close_plots,

        )

    # ------------------------------------------------------
    # 9. Completion summary
    # ------------------------------------------------------

    feasible_count = sum(

        dp is not None

        for dp in design_points.values()

    )

    total_count = len(design_points)

    print(SEP)

    print("CONSTRAINT ANALYSIS COMPLETED")

    print(
        f"Clean CL,max at AR {float(AR[PLOT_AR_INDEX, 0]):.2f}: "
        f"{float(clean_stall_ws_sweep[PLOT_AR_INDEX, 0]) / p['dynamic_pressure_stall_pa']:.3f}"
    )

    print(
        f"Required CL,max at stall W/S limit "
        f"({ws_limits['stall']:.1f} N/m²): "
        f"{cl_max_required_at_stall_limit:.3f}"
    )

    print(f"AR points: {AR.shape[0]}")

    print(f"Velocity points: {V.shape[1]}")

    print(f"W/S points: {WS_VALUES.size}")

    print(
        f"Feasible combinations: "
        f"{feasible_count}/{total_count}"
    )

    print(f"Output directory: {OUTPUT_DIR}")

    print(SEP)

    return {
        "parameters": p,
        "AR": AR,
        "V": V,
        "design_points": design_points,
        "ws_selected": ws_selected,
        "pw_selected": pw_selected,
        "tw": tw,
        "pw_curves": pw_curves,
        "ws_limits": ws_limits,
        "q_cruise_grid": q_cruise_grid,
        "units": units,
    }


# ==========================================================
# SCRIPT EXECUTION
# ==========================================================

if __name__ == "__main__":
    main()
