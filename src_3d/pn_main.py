import argparse

import matplotlib.pyplot as plt
import numpy as np

from missile import Missile
from pn import ProportionalNavigation
from sim import Sim
from vehicle import Vehicle


def create_missile(target, speed=300.0, max_g=20.0):
    position = np.zeros(3)
    relative_position = target.get_state()[0:3] - position

    return Missile(
        x=position[0],
        y=position[1],
        z=position[2],
        v=speed,
        flight_path_angle=np.arctan2(
            relative_position[2],
            np.hypot(relative_position[0], relative_position[1]),
        ),
        heading=np.arctan2(relative_position[1], relative_position[0]),
        max_g=max_g,
    )


def step_minimum_distance(previous_relative, current_relative):
    relative_change = current_relative - previous_relative
    change_squared = np.dot(relative_change, relative_change)

    if change_squared <= 0.0:
        return float(np.linalg.norm(previous_relative)), 0.0

    fraction = np.clip(
        -np.dot(previous_relative, relative_change) / change_squared,
        0.0,
        1.0,
    )
    closest_relative = previous_relative + fraction * relative_change

    return float(np.linalg.norm(closest_relative)), float(fraction)


def run_simulation(
    seed=103142,
    dt=0.05,
    max_time=45.0,
    intercept_radius=3.0,
    navigation_constant=3.0,
    max_g=20.0,
):
    target = Vehicle(
        x=-2000.0,
        y=-1500.0,
        z=1200.0,
        v=150.0,
        flight_path_angle=np.radians(5.0),
        heading=np.radians(80.0),
    )
    missile = create_missile(target, speed=300.0, max_g=max_g)
    guidance = ProportionalNavigation(
        navigation_constant=navigation_constant,
        max_g=max_g,
    )

    maneuver = Sim(
        target=target,
        dt=dt,
        T_max=max_time,
        seed=seed,
        intercept_radius=intercept_radius,
        maneuver_profile="aggressive",
    )
    maneuver.select_new_move(0.0)

    target_trajectory = [target.get_state()]
    missile_trajectory = [missile.get_state()]
    control_history = []
    distance_history = [
        np.linalg.norm(target.get_state()[0:3] - missile.get_state()[0:3])
    ]

    intercepted = False
    intercept_time = None
    minimum_distance = float(distance_history[0])
    step_count = 0

    while step_count * dt < max_time:
        time = step_count * dt

        if time >= maneuver.move_end_time:
            maneuver.select_new_move(time)

        target_command = maneuver.get_command(time)
        missile_command = guidance.solve(
            missile.get_state(),
            target.get_state(),
        )
        previous_relative = target.get_state()[0:3] - missile.get_state()[0:3]

        target.set_acceleration_cmd(target_command)
        missile.update_acceleration(missile_command)
        target.step(dt)
        missile.step(dt)

        current_relative = target.get_state()[0:3] - missile.get_state()[0:3]
        distance, fraction = step_minimum_distance(
            previous_relative,
            current_relative,
        )

        target_trajectory.append(target.get_state())
        missile_trajectory.append(missile.get_state())
        control_history.append(missile_command.copy())
        distance_history.append(np.linalg.norm(current_relative))
        minimum_distance = min(minimum_distance, distance)

        if distance <= intercept_radius:
            intercepted = True
            intercept_time = time + fraction * dt
            break

        step_count += 1

    return {
        "dt": float(dt),
        "seed": int(seed),
        "intercepted": intercepted,
        "intercept_time": intercept_time,
        "minimum_distance": minimum_distance,
        "target_trajectory": np.asarray(target_trajectory),
        "missile_trajectory": np.asarray(missile_trajectory),
        "control_history": np.asarray(control_history),
        "distance_history": np.asarray(distance_history),
    }


def plot_result(result):
    target = result["target_trajectory"]
    missile = result["missile_trajectory"]
    time = np.arange(len(result["distance_history"])) * result["dt"]

    figure = plt.figure(figsize=(13, 5.5))
    trajectory_axis = figure.add_subplot(121, projection="3d")
    distance_axis = figure.add_subplot(122)

    trajectory_axis.plot(
        missile[:, 0], missile[:, 1], missile[:, 2], label="PN pursuer"
    )
    trajectory_axis.plot(
        target[:, 0], target[:, 1], target[:, 2], label="Target"
    )
    trajectory_axis.set_xlabel("X [m]")
    trajectory_axis.set_ylabel("Y [m]")
    trajectory_axis.set_zlabel("Z [m]")
    trajectory_axis.set_title("3D proportional navigation")
    trajectory_axis.legend()

    distance_axis.plot(time, result["distance_history"])
    distance_axis.axhline(3.0, color="tab:red", linestyle="--", label="3 m hit radius")
    distance_axis.set_xlabel("Time [s]")
    distance_axis.set_ylabel("Distance [m]")
    distance_axis.set_title("Target-pursuer distance")
    distance_axis.grid(alpha=0.25)
    distance_axis.legend()

    figure.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run the 3D PN baseline.")
    parser.add_argument("--seed", type=int, default=103142)
    parser.add_argument("--navigation-constant", type=float, default=3.0)
    parser.add_argument("--max-time", type=float, default=45.0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    result = run_simulation(
        seed=args.seed,
        navigation_constant=args.navigation_constant,
        max_time=args.max_time,
    )

    intercept_time = result["intercept_time"]
    intercept_text = "--" if intercept_time is None else f"{intercept_time:.3f} s"

    print(f"Seed: {result['seed']}")
    print(f"Intercepted: {result['intercepted']}")
    print(f"Minimum distance: {result['minimum_distance']:.3f} m")
    print(f"Intercept time: {intercept_text}")

    if args.plot:
        plot_result(result)


if __name__ == "__main__":
    main()
