import pickle
import random
from collections import deque
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np


class Sim:

    def __init__(
        self,
        target,
        missile=None,
        dt=0.05,
        T_max=15.0,
        seed=None,
        predictor=None,
        mpc=None,
        observation_time=2.0,
        intercept_radius=3.0,
        maneuver_profile="standard",
    ):
        self.target = target
        self.missile = missile
        self.dt = float(dt)
        self.T_max = float(T_max)
        self.rng = random.Random(seed)
        self.predictor = predictor
        self.mpc = mpc
        self.observation_time = float(observation_time)
        self.intercept_radius = float(intercept_radius)
        self.maneuver_profile = maneuver_profile

        self.current_mode = None
        self.move_start_time = 0.0
        self.move_end_time = 0.0

        self.parallel_acceleration = 0.0
        self.vertical_acceleration = 0.0
        self.side_acceleration = 0.0

        self.vertical_period = 4.0
        self.side_period = 4.0
        self.vertical_phase = 0.0
        self.side_phase = 0.0

        self.command_history = []
        self.mode_history = []
        self.prediction_origins = []

    def select_new_move(self, time):
        self.move_start_time = float(time)

        modes = ["straight", "speed_change", "turn", "climb", "weave"]

        if self.maneuver_profile == "aggressive":
            weights = [0.02, 0.04, 0.17, 0.12, 0.65]
        elif self.maneuver_profile == "dynamic":
            weights = [0.05, 0.10, 0.20, 0.15, 0.50]
        else:
            weights = [0.25, 0.15, 0.20, 0.15, 0.25]

        self.current_mode = self.rng.choices(
            population=modes,
            weights=weights,
            k=1,
        )[0]

        self.parallel_acceleration = 0.0
        self.vertical_acceleration = 0.0
        self.side_acceleration = 0.0
        self.vertical_phase = 0.0
        self.side_phase = 0.0

        if self.current_mode == "straight":
            if self.maneuver_profile == "aggressive":
                duration = self.rng.uniform(0.7, 1.4)
            elif self.maneuver_profile == "dynamic":
                duration = self.rng.uniform(0.75, 1.5)
            else:
                duration = self.rng.uniform(2.0, 4.0)

        elif self.current_mode == "speed_change":
            direction = self.rng.choice([-1.0, 1.0])
            self.parallel_acceleration = direction * self.rng.uniform(
                0.05 * self.target.g,
                0.30 * self.target.g,
            )
            duration = self.rng.uniform(1.0, 3.0)

        elif self.current_mode == "turn":
            direction = self.rng.choice([-1.0, 1.0])

            if self.maneuver_profile == "aggressive":
                self.side_acceleration = direction * self.rng.uniform(
                    3.0 * self.target.g,
                    6.5 * self.target.g,
                )
                self.vertical_acceleration = self.rng.uniform(
                    -3.0 * self.target.g,
                    3.0 * self.target.g,
                )
                self.move_end_time = self.move_start_time + self.rng.uniform(0.7, 1.4)
                return

            self.side_acceleration = direction * self.rng.uniform(
                1.5 * self.target.g,
                5.0 * self.target.g,
            )

            if self.maneuver_profile == "dynamic":
                self.vertical_acceleration = self.rng.uniform(
                    -1.5 * self.target.g,
                    1.5 * self.target.g,
                )
                duration = self.rng.uniform(1.0, 2.5)
            else:
                duration = self.rng.uniform(2.0, 4.0)

        elif self.current_mode == "climb":
            direction = self.rng.choice([-1.0, 1.0])

            if self.maneuver_profile == "aggressive":
                self.vertical_acceleration = direction * self.rng.uniform(
                    2.5 * self.target.g,
                    5.0 * self.target.g,
                )
                self.side_acceleration = self.rng.uniform(
                    -2.5 * self.target.g,
                    2.5 * self.target.g,
                )
                self.move_end_time = self.move_start_time + self.rng.uniform(0.7, 1.4)
                return

            self.vertical_acceleration = direction * self.rng.uniform(
                1.0 * self.target.g,
                3.5 * self.target.g,
            )

            if self.maneuver_profile == "dynamic":
                self.side_acceleration = self.rng.uniform(
                    -1.0 * self.target.g,
                    1.0 * self.target.g,
                )
                duration = self.rng.uniform(1.0, 2.5)
            else:
                duration = self.rng.uniform(2.0, 4.0)

        elif self.current_mode == "weave":
            if self.maneuver_profile == "aggressive":
                side_range = (3.5, 6.5)
                vertical_range = (3.0, 5.5)
            else:
                side_range = (2.0, 5.5)
                vertical_range = (1.0, 4.0)

            self.side_acceleration = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(
                side_range[0] * self.target.g,
                side_range[1] * self.target.g,
            )
            self.vertical_acceleration = self.rng.choice(
                [-1.0, 1.0]
            ) * self.rng.uniform(
                vertical_range[0] * self.target.g,
                vertical_range[1] * self.target.g,
            )

            if self.maneuver_profile == "aggressive":
                self.side_period = self.rng.uniform(1.8, 2.8)
                self.vertical_period = self.rng.uniform(2.0, 3.0)
                n_cycles = self.rng.randint(2, 3)
            elif self.maneuver_profile == "dynamic":
                self.side_period = self.rng.uniform(3.0, 5.0)
                self.vertical_period = self.rng.uniform(3.5, 5.5)
                n_cycles = self.rng.randint(1, 2)
            else:
                self.side_period = self.rng.uniform(5.0, 8.0)
                self.vertical_period = self.rng.uniform(5.0, 8.0)
                n_cycles = 1

            self.side_phase = self.rng.uniform(0.0, 2.0 * np.pi)
            self.vertical_phase = self.side_phase + self.rng.choice(
                [-0.5 * np.pi, 0.5 * np.pi]
            )
            duration = n_cycles * max(
                self.side_period,
                self.vertical_period,
            )

        self.move_end_time = self.move_start_time + duration

    def get_command(self, time):
        maneuver_time = float(time) - self.move_start_time

        parallel_command = self.parallel_acceleration
        vertical_command = self.vertical_acceleration
        side_command = self.side_acceleration

        if self.current_mode == "weave":
            side_wave = np.sin(
                2.0 * np.pi * maneuver_time / self.side_period + self.side_phase
            )
            vertical_wave = np.sin(
                2.0 * np.pi * maneuver_time / self.vertical_period + self.vertical_phase
            )

            if self.maneuver_profile in {"dynamic", "aggressive"}:
                side_wave = np.tanh(2.0 * side_wave)
                vertical_wave = np.tanh(2.0 * vertical_wave)

            side_command *= side_wave
            vertical_command *= vertical_wave

        return np.array(
            [
                parallel_command,
                vertical_command,
                side_command,
            ]
        )

    def dataset_sim(self):
        dataset = [self.target.get_pos_vel()]
        self.command_history = []
        self.mode_history = []

        time = 0.0
        step_count = 0
        self.select_new_move(time)

        while time < self.T_max:
            if time >= self.move_end_time:
                self.select_new_move(time)

            acceleration_command = self.get_command(time)

            self.target.set_acceleration_cmd(acceleration_command)
            self.target.step(self.dt)

            dataset.append(self.target.get_pos_vel())
            self.command_history.append(acceleration_command.copy())
            self.mode_history.append(self.current_mode)

            step_count += 1
            time = step_count * self.dt

        return np.asarray(dataset)

    @staticmethod
    def direction(flight_path_angle, heading):
        cos_gamma = np.cos(flight_path_angle)

        return np.array(
            [
                cos_gamma * np.cos(heading),
                cos_gamma * np.sin(heading),
                np.sin(flight_path_angle),
            ]
        )

    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def predict_positions(self, target_state, predicted_accelerations):
        target_state = np.asarray(target_state, dtype=float)
        predicted_accelerations = np.asarray(
            predicted_accelerations,
            dtype=float,
        )

        position = target_state[0:3].copy()
        speed = target_state[3]
        gamma = target_state[4]
        heading = target_state[5]

        predicted_positions = []

        for acceleration in predicted_accelerations:
            parallel_acceleration = acceleration[0]
            vertical_acceleration = acceleration[1]
            side_acceleration = acceleration[2]

            safe_speed = max(speed, 1e-6)
            safe_cos = max(abs(np.cos(gamma)), 1e-3)

            speed = max(
                speed + parallel_acceleration * self.dt,
                1e-6,
            )

            gamma += vertical_acceleration / safe_speed * self.dt

            heading += side_acceleration / (safe_speed * safe_cos) * self.dt
            heading = self.wrap_angle(heading)

            velocity = speed * self.direction(gamma, heading)
            position += velocity * self.dt

            predicted_positions.append(position.copy())

        return np.asarray(predicted_positions)

    def simulation(self):
        if self.missile is None:
            raise ValueError("missile is required for simulation().")
        if self.predictor is None:
            raise ValueError("predictor is required for simulation().")
        if self.mpc is None:
            raise ValueError("mpc is required for simulation().")

        target_trajectory = []
        missile_trajectory = []
        predictions = []

        history = deque(maxlen=40)
        history.append(self.target.get_pos_vel())

        ## LSTM 입력에 필요한 과거 2초를 먼저 관측
        observation_steps = int(np.ceil(self.observation_time / self.dt))
        observation_start = -observation_steps * self.dt
        self.select_new_move(observation_start)

        for step in range(observation_steps):
            observation_time = observation_start + step * self.dt

            if observation_time >= self.move_end_time:
                self.select_new_move(observation_time)

            acceleration_command = self.get_command(observation_time)

            self.target.set_acceleration_cmd(acceleration_command)
            self.target.step(self.dt)

            history.append(self.target.get_pos_vel())

        target_trajectory.append(self.target.get_state())
        missile_trajectory.append(self.missile.get_state())

        self.prediction_origins = []
        self.command_history = []
        self.mode_history = []

        step_count = 0

        while step_count * self.dt < self.T_max:
            time = step_count * self.dt

            if time >= self.move_end_time:
                self.select_new_move(time)

            target_command = self.get_command(time)

            predicted_accelerations = self.predictor.predict(np.asarray(history))

            pursuer_command = self.mpc.solve(
                self.missile.get_state(),
                self.target.get_state(),
                predicted_accelerations,
            )

            predicted_positions = self.predict_positions(
                self.target.get_state(),
                predicted_accelerations,
            )

            predictions.append(predicted_positions)
            self.prediction_origins.append(self.target.get_state()[0:3].copy())

            self.missile.update_acceleration(pursuer_command)

            previous_relative_position = (
                self.target.get_state()[0:3] - self.missile.get_state()[0:3]
            )

            self.target.set_acceleration_cmd(target_command)
            self.target.step(self.dt)
            self.missile.step(self.dt)

            target_trajectory.append(self.target.get_state())
            missile_trajectory.append(self.missile.get_state())

            history.append(self.target.get_pos_vel())
            self.command_history.append(target_command.copy())
            self.mode_history.append(self.current_mode)

            current_relative_position = (
                self.target.get_state()[0:3] - self.missile.get_state()[0:3]
            )

            relative_change = current_relative_position - previous_relative_position

            change_squared = np.dot(
                relative_change,
                relative_change,
            )

            if change_squared > 0.0:
                fraction = np.clip(
                    -np.dot(
                        previous_relative_position,
                        relative_change,
                    )
                    / change_squared,
                    0.0,
                    1.0,
                )
            else:
                fraction = 0.0

            closest_relative_position = (
                previous_relative_position + fraction * relative_change
            )

            step_minimum_distance = np.linalg.norm(closest_relative_position)

            if step_minimum_distance <= self.intercept_radius:
                break

            step_count += 1

        return (
            np.asarray(missile_trajectory),
            np.asarray(target_trajectory),
            predictions,
        )

    def plot_trajectory(
        self,
        missile_trajectory,
        target_trajectory,
        predictions=None,
    ):
        missile_trajectory = np.asarray(
            missile_trajectory,
            dtype=float,
        )
        target_trajectory = np.asarray(
            target_trajectory,
            dtype=float,
        )

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            missile_trajectory[:, 0],
            missile_trajectory[:, 1],
            missile_trajectory[:, 2],
            color="#FF7A45",
            linewidth=2.0,
            label="Pursuer",
        )

        ax.plot(
            target_trajectory[:, 0],
            target_trajectory[:, 1],
            target_trajectory[:, 2],
            color="#35D0BA",
            linewidth=2.5,
            label="Target",
        )

        ## 약 1초 간격으로 LSTM의 전체 0.4초 예측경로를 표시
        if predictions:
            prediction_interval = max(
                int(round(1.0 / self.dt)),
                1,
            )

            for index in range(
                0,
                len(predictions),
                prediction_interval,
            ):
                prediction_path = np.vstack(
                    (
                        self.prediction_origins[index],
                        predictions[index],
                    )
                )

                label = "LSTM 0.4 s prediction" if index == 0 else None

                ax.plot(
                    prediction_path[:, 0],
                    prediction_path[:, 1],
                    prediction_path[:, 2],
                    color="#1F2933",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.75,
                    label=label,
                )

        ax.scatter(
            missile_trajectory[0, 0],
            missile_trajectory[0, 1],
            missile_trajectory[0, 2],
            color="#FF7A45",
            s=55,
        )

        ax.scatter(
            target_trajectory[0, 0],
            target_trajectory[0, 1],
            target_trajectory[0, 2],
            color="#35D0BA",
            s=55,
        )

        ax.scatter(
            missile_trajectory[-1, 0],
            missile_trajectory[-1, 1],
            missile_trajectory[-1, 2],
            color="#FF7A45",
            marker="x",
            s=75,
        )

        ax.scatter(
            target_trajectory[-1, 0],
            target_trajectory[-1, 1],
            target_trajectory[-1, 2],
            color="#35D0BA",
            marker="x",
            s=75,
        )

        all_positions = np.vstack(
            (
                missile_trajectory[:, 0:3],
                target_trajectory[:, 0:3],
            )
        )
        ranges = np.ptp(all_positions, axis=0)
        ax.set_box_aspect(np.maximum(ranges, 1.0))

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("3D LSTM-MPC Simulation")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()

        plt.show()

    def plot_target_trajectory(self, dataset):
        dataset = np.asarray(dataset, dtype=float)

        if dataset.ndim != 2 or dataset.shape[1] != 6:
            raise ValueError("dataset must have shape (steps, 6).")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            dataset[:, 0],
            dataset[:, 1],
            dataset[:, 2],
            color="#35D0BA",
            linewidth=2.2,
            label="Target",
        )

        ax.scatter(
            dataset[0, 0],
            dataset[0, 1],
            dataset[0, 2],
            color="#2563EB",
            s=55,
            label="Start",
        )

        ax.scatter(
            dataset[-1, 0],
            dataset[-1, 1],
            dataset[-1, 2],
            color="#EF4444",
            marker="x",
            s=70,
            label="End",
        )

        ranges = np.ptp(dataset[:, 0:3], axis=0)
        ax.set_box_aspect(np.maximum(ranges, 1.0))

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("3D Target Maneuver")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()

        plt.show()


class ComparisonSim(Sim):
    """Run both guidance methods against one shared target trajectory."""

    def __init__(
        self,
        target,
        acceleration_missile,
        position_missile,
        acceleration_predictor,
        position_predictor,
        acceleration_mpc,
        position_mpc,
        dt=0.05,
        T_max=45.0,
        seed=None,
        observation_time=2.0,
        intercept_radius=3.0,
        maneuver_profile="dynamic",
    ):
        super().__init__(
            target=target,
            dt=dt,
            T_max=T_max,
            seed=seed,
            observation_time=observation_time,
            intercept_radius=intercept_radius,
            maneuver_profile=maneuver_profile,
        )

        self.acceleration_missile = acceleration_missile
        self.position_missile = position_missile

        self.acceleration_predictor = acceleration_predictor
        self.position_predictor = position_predictor

        self.acceleration_mpc = acceleration_mpc
        self.position_mpc = position_mpc

    @staticmethod
    def step_minimum_distance(
        previous_relative_position,
        current_relative_position,
    ):
        relative_change = current_relative_position - previous_relative_position

        change_squared = np.dot(
            relative_change,
            relative_change,
        )

        if change_squared > 0.0:
            fraction = np.clip(
                -np.dot(
                    previous_relative_position,
                    relative_change,
                )
                / change_squared,
                0.0,
                1.0,
            )
        else:
            fraction = 0.0

        closest_relative_position = (
            previous_relative_position + fraction * relative_change
        )

        return (
            float(np.linalg.norm(closest_relative_position)),
            float(fraction),
        )

    def observe_target(self, history):
        observation_steps = int(np.ceil(self.observation_time / self.dt))
        observation_start = -observation_steps * self.dt
        self.select_new_move(observation_start)

        for step in range(observation_steps):
            time = observation_start + step * self.dt

            if time >= self.move_end_time:
                self.select_new_move(time)

            command = self.get_command(time)
            self.target.set_acceleration_cmd(command)
            self.target.step(self.dt)
            history.append(self.target.get_pos_vel())

    def method_result(
        self,
        trajectory,
        controls,
        predictions,
        origins,
        intercepted,
        intercept_time,
        minimum_distance,
        runtime,
        extra=None,
    ):
        result = {
            "missile_trajectory": np.asarray(trajectory),
            "control_history": np.asarray(controls),
            "predictions": predictions,
            "prediction_origins": origins,
            "intercepted": bool(intercepted),
            "intercept_time": intercept_time,
            "minimum_distance": float(minimum_distance),
            "runtime": float(runtime),
        }

        if extra is not None:
            result.update(extra)

        return result

    def simulation(self):
        history = deque(maxlen=40)
        history.append(self.target.get_pos_vel())
        self.observe_target(history)

        target_trajectory = [self.target.get_state()]

        acceleration_trajectory = [self.acceleration_missile.get_state()]
        position_trajectory = [self.position_missile.get_state()]

        acceleration_controls = []
        position_controls = []

        acceleration_predictions = []
        position_predictions = []
        acceleration_origins = []
        position_origins = []
        position_hit_steps = []
        position_hit_times = []
        position_hit_points = []
        position_hit_reachable = []

        acceleration_intercepted = False
        position_intercepted = False
        acceleration_intercept_time = None
        position_intercept_time = None

        acceleration_minimum_distance = np.linalg.norm(
            self.target.get_state()[0:3] - self.acceleration_missile.get_state()[0:3]
        )
        position_minimum_distance = np.linalg.norm(
            self.target.get_state()[0:3] - self.position_missile.get_state()[0:3]
        )

        acceleration_runtime = 0.0
        position_runtime = 0.0

        self.command_history = []
        self.mode_history = []
        step_count = 0

        while step_count * self.dt < self.T_max:
            time = step_count * self.dt

            if time >= self.move_end_time:
                self.select_new_move(time)

            target_command = self.get_command(time)
            history_array = np.asarray(history)
            target_state = self.target.get_state()
            target_position = target_state[0:3]

            if acceleration_intercepted:
                predicted_accelerations = None
                acceleration_positions = None
            else:
                predicted_accelerations = self.acceleration_predictor.predict(
                    history_array
                )
                acceleration_positions = self.predict_positions(
                    target_state,
                    predicted_accelerations,
                )

            if position_intercepted:
                predicted_positions = None
            else:
                predicted_positions = self.position_predictor.predict(
                    history_array
                )

            acceleration_predictions.append(acceleration_positions)
            position_predictions.append(predicted_positions)
            acceleration_origins.append(target_position.copy())
            position_origins.append(target_position.copy())

            previous_acceleration_relative = (
                target_position - self.acceleration_missile.get_state()[0:3]
            )
            previous_position_relative = (
                target_position - self.position_missile.get_state()[0:3]
            )

            if not acceleration_intercepted:
                start = perf_counter()
                acceleration_command = self.acceleration_mpc.solve(
                    self.acceleration_missile.get_state(),
                    target_state,
                    predicted_accelerations,
                )
                acceleration_runtime += perf_counter() - start
                self.acceleration_missile.update_acceleration(acceleration_command)
                acceleration_controls.append(np.asarray(acceleration_command).copy())

            if not position_intercepted:
                start = perf_counter()
                position_command = self.position_mpc.solve(
                    self.position_missile.get_state(),
                    predicted_positions,
                )
                position_runtime += perf_counter() - start
                self.position_missile.update_acceleration(position_command)
                position_controls.append(np.asarray(position_command).copy())

                estimate = self.position_mpc.last_info["estimate"]
                position_hit_steps.append(int(estimate.j_hit))
                position_hit_times.append(float(estimate.t_hit))
                position_hit_points.append(
                    np.asarray(estimate.target_position).copy()
                )
                position_hit_reachable.append(bool(estimate.reachable))
            else:
                position_hit_steps.append(None)
                position_hit_times.append(None)
                position_hit_points.append(None)
                position_hit_reachable.append(None)

            self.target.set_acceleration_cmd(target_command)
            self.target.step(self.dt)

            if not acceleration_intercepted:
                self.acceleration_missile.step(self.dt)

            if not position_intercepted:
                self.position_missile.step(self.dt)

            target_trajectory.append(self.target.get_state())
            acceleration_trajectory.append(self.acceleration_missile.get_state())
            position_trajectory.append(self.position_missile.get_state())

            history.append(self.target.get_pos_vel())
            self.command_history.append(target_command.copy())
            self.mode_history.append(self.current_mode)

            current_target_position = self.target.get_state()[0:3]

            if not acceleration_intercepted:
                current_relative = (
                    current_target_position - self.acceleration_missile.get_state()[0:3]
                )
                distance, fraction = self.step_minimum_distance(
                    previous_acceleration_relative,
                    current_relative,
                )
                acceleration_minimum_distance = min(
                    acceleration_minimum_distance,
                    distance,
                )

                if distance <= self.intercept_radius:
                    acceleration_intercepted = True
                    acceleration_intercept_time = time + fraction * self.dt

            if not position_intercepted:
                current_relative = (
                    current_target_position - self.position_missile.get_state()[0:3]
                )
                distance, fraction = self.step_minimum_distance(
                    previous_position_relative,
                    current_relative,
                )
                position_minimum_distance = min(
                    position_minimum_distance,
                    distance,
                )

                if distance <= self.intercept_radius:
                    position_intercepted = True
                    position_intercept_time = time + fraction * self.dt

            step_count += 1

            if acceleration_intercepted and position_intercepted:
                break

        target_trajectory = np.asarray(target_trajectory)

        return {
            "target_trajectory": target_trajectory,
            "target_control_history": np.asarray(self.command_history),
            "target_mode_history": list(self.mode_history),
            "dt": self.dt,
            "intercept_radius": self.intercept_radius,
            "acceleration": self.method_result(
                acceleration_trajectory,
                acceleration_controls,
                acceleration_predictions,
                acceleration_origins,
                acceleration_intercepted,
                acceleration_intercept_time,
                acceleration_minimum_distance,
                acceleration_runtime,
            ),
            "position": self.method_result(
                position_trajectory,
                position_controls,
                position_predictions,
                position_origins,
                position_intercepted,
                position_intercept_time,
                position_minimum_distance,
                position_runtime,
                extra={
                    "hit_steps": position_hit_steps,
                    "hit_times": position_hit_times,
                    "hit_points": position_hit_points,
                    "hit_reachable": position_hit_reachable,
                },
            ),
        }


def save_comparison_result(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("wb") as file:
        pickle.dump(
            result,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return output_path


def load_comparison_result(input_path):
    input_path = Path(input_path)

    with input_path.open("rb") as file:
        return pickle.load(file)
