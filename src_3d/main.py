from pathlib import Path

import numpy as np

from missile import Missile
from model import (
    AccelerationTargetPredictor,
    PositionTargetPredictor,
)
from mpc import (
    AccelerationDisturbanceMPC,
    PositionInterceptionMPC,
)
from sim import (
    ComparisonSim,
    load_comparison_result,
    save_comparison_result,
)
from vehicle import Vehicle
from visualization import animate_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_missile(target, speed=300.0, max_g=20.0):
    missile_position = np.array(
        [
            0.0,
            0.0,
            0.0,
        ]
    )

    relative_position = target.get_state()[0:3] - missile_position

    initial_heading = np.arctan2(
        relative_position[1],
        relative_position[0],
    )

    initial_flight_path_angle = np.arctan2(
        relative_position[2],
        np.hypot(
            relative_position[0],
            relative_position[1],
        ),
    )

    return Missile(
        x=missile_position[0],
        y=missile_position[1],
        z=missile_position[2],
        v=speed,
        flight_path_angle=initial_flight_path_angle,
        heading=initial_heading,
        max_g=max_g,
    )


def print_method_result(title, result, solver_failures):
    intercept_time = result["intercept_time"]

    if intercept_time is None:
        intercept_time_text = "--"
    else:
        intercept_time_text = f"{intercept_time:.3f} s"

    print()
    print(title)
    print(f"Intercepted: {result['intercepted']}")
    print(f"Minimum distance: " f"{result['minimum_distance']:.3f} m")
    print(f"Intercept time: {intercept_time_text}")
    print(f"MPC runtime: {result['runtime']:.3f} s")
    print(f"QP failures: {solver_failures}")


def run_comparison(seed):
    dt = 0.05
    intercept_radius = 3.0

    target = Vehicle(
        x=-2000.0,
        y=-1500.0,
        z=1200.0,
        v=150.0,
        flight_path_angle=np.radians(5.0),
        heading=np.radians(80.0),
    )

    acceleration_missile = create_missile(target)
    position_missile = create_missile(target)

    acceleration_predictor = AccelerationTargetPredictor()
    position_predictor = PositionTargetPredictor()

    acceleration_mpc = AccelerationDisturbanceMPC(
        dt=dt,
        horizon=20,
        disturbance_steps=8,
        max_g=20.0,
    )

    position_mpc = PositionInterceptionMPC(
        dt=dt,
        max_g=20.0,
        intercept_radius=intercept_radius,
        max_horizon=100,
        control_block_size=5,
    )

    sim = ComparisonSim(
        target=target,
        acceleration_missile=acceleration_missile,
        position_missile=position_missile,
        acceleration_predictor=acceleration_predictor,
        position_predictor=position_predictor,
        acceleration_mpc=acceleration_mpc,
        position_mpc=position_mpc,
        dt=dt,
        T_max=45.0,
        seed=seed,
        observation_time=2.0,
        intercept_radius=intercept_radius,
        maneuver_profile="aggressive",
    )

    result = sim.simulation()

    result["seed"] = seed
    result["maneuver_profile"] = "aggressive"
    result["target_speed"] = 150.0
    result["pursuer_speed"] = 300.0
    result["acceleration"]["solver_failures"] = acceleration_mpc.solver_failures
    result["position"]["solver_failures"] = position_mpc.solver_failures
    result["position"]["qp_solve_count"] = position_mpc.qp_solve_count

    return result


def main():
    seed = 103142
    recompute = True
    result_path = (
        PROJECT_ROOT
        / "results_3d"
        / "seed_runs"
        / f"ab_comparison_aggressive_v300_seed{seed}.pkl"
    )

    if result_path.exists() and not recompute:
        result = load_comparison_result(result_path)
        print(f"Loaded saved result: {result_path}")
    else:
        result = run_comparison(seed)
        save_comparison_result(result, result_path)
        print(f"Saved result: {result_path}")

    simulation_time = (len(result["target_trajectory"]) - 1) * result["dt"]

    print(f"Shared simulation time: {simulation_time:.2f} s")
    print(f"Shared target maneuver seed: {result['seed']}")

    print_method_result(
        "A. Acceleration disturbance MPC",
        result["acceleration"],
        result["acceleration"]["solver_failures"],
    )

    print_method_result(
        "B. Position interception MPC",
        result["position"],
        result["position"]["solver_failures"],
    )

    animate_comparison(
        result=result,
        playback_speed=1.0,
        frame_stride=1,
    )


if __name__ == "__main__":
    main()
