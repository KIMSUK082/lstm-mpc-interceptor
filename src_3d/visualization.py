from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np


class InterceptionAnimation:
    def __init__(
        self,
        missile_trajectory,
        target_trajectory,
        predictions=None,
        prediction_origins=None,
        dt=0.05,
        playback_speed=1.0,
        intercept_radius=3.0,
        frame_stride=1,
    ):
        self.missile = np.asarray(missile_trajectory, dtype=float)
        self.target = np.asarray(target_trajectory, dtype=float)
        self.predictions = [] if predictions is None else predictions
        self.prediction_origins = (
            [] if prediction_origins is None else prediction_origins
        )
        self.dt = float(dt)
        self.playback_speed = float(playback_speed)
        self.intercept_radius = float(intercept_radius)
        self.frame_stride = int(frame_stride)

        self.validate_inputs()
        self.frames = list(range(0, len(self.missile), self.frame_stride))
        if self.frames[-1] != len(self.missile) - 1:
            self.frames.append(len(self.missile) - 1)

        self.figure = plt.figure(figsize=(12, 8))
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.configure_axis()
        self.create_artists()
        self.animation = None

    def validate_inputs(self):
        if self.missile.ndim != 2 or self.missile.shape[1] < 3:
            raise ValueError("missile_trajectory must have shape (steps, 3 or more).")
        if self.target.ndim != 2 or self.target.shape[1] < 3:
            raise ValueError("target_trajectory must have shape (steps, 3 or more).")
        if len(self.missile) != len(self.target):
            raise ValueError("missile and target trajectories must have equal length.")
        if len(self.missile) == 0:
            raise ValueError("trajectories must not be empty.")
        if self.dt <= 0.0:
            raise ValueError("dt must be greater than zero.")
        if self.playback_speed <= 0.0:
            raise ValueError("playback_speed must be greater than zero.")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be at least one.")

    @staticmethod
    def padded_limits(values, padding_ratio=0.08):
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        span = maximum - minimum
        padding = max(span * padding_ratio, 10.0)
        return minimum - padding, maximum + padding

    def configure_axis(self):
        positions = np.vstack((self.missile[:, :3], self.target[:, :3]))
        x_limits = self.padded_limits(positions[:, 0])
        y_limits = self.padded_limits(positions[:, 1])
        z_limits = self.padded_limits(positions[:, 2])

        self.axis.set_xlim(x_limits)
        self.axis.set_ylim(y_limits)
        self.axis.set_zlim(z_limits)
        self.axis.set_box_aspect(
            np.maximum(
                [
                    x_limits[1] - x_limits[0],
                    y_limits[1] - y_limits[0],
                    z_limits[1] - z_limits[0],
                ],
                1.0,
            )
        )
        self.axis.set_xlabel("X [m]")
        self.axis.set_ylabel("Y [m]")
        self.axis.set_zlabel("Z [m]")
        self.axis.set_title("3D LSTM-MPC Interception")
        self.axis.grid(alpha=0.25)
        self.axis.view_init(elev=24.0, azim=-58.0)

    def create_artists(self):
        (self.missile_path,) = self.axis.plot(
            [], [], [], color="#FF7043", linewidth=2.2, label="Pursuer"
        )
        (self.target_path,) = self.axis.plot(
            [], [], [], color="#26C6B8", linewidth=2.4, label="Target"
        )
        (self.prediction_path,) = self.axis.plot(
            [],
            [],
            [],
            color="#263238",
            linestyle="--",
            linewidth=1.6,
            label="LSTM 0.4 s prediction",
        )
        (self.relative_line,) = self.axis.plot(
            [], [], [], color="#7E8A97", linewidth=1.0, alpha=0.65
        )
        (self.missile_marker,) = self.axis.plot(
            [], [], [], marker="o", color="#FF7043", markersize=7
        )
        (self.target_marker,) = self.axis.plot(
            [], [], [], marker="o", color="#26C6B8", markersize=7
        )
        self.status_text = self.axis.text2D(
            0.02,
            0.96,
            "",
            transform=self.axis.transAxes,
            va="top",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "white",
                "edgecolor": "#C8D0D8",
                "alpha": 0.92,
            },
        )
        self.axis.legend(loc="upper right")
        self.figure.tight_layout()

    @staticmethod
    def set_line(line, points):
        points = np.asarray(points, dtype=float)
        line.set_data(points[:, 0], points[:, 1])
        line.set_3d_properties(points[:, 2])

    def prediction_for_frame(self, frame):
        if frame >= len(self.predictions):
            return None

        prediction = np.asarray(self.predictions[frame], dtype=float)
        if prediction.ndim != 2 or prediction.shape[1] < 3:
            return None

        if frame < len(self.prediction_origins):
            origin = np.asarray(self.prediction_origins[frame], dtype=float)[:3]
        else:
            origin = self.target[frame, :3]
        return np.vstack((origin, prediction[:, :3]))

    def update(self, frame):
        missile_position = self.missile[frame, :3]
        target_position = self.target[frame, :3]
        distance = float(np.linalg.norm(target_position - missile_position))

        self.set_line(self.missile_path, self.missile[: frame + 1, :3])
        self.set_line(self.target_path, self.target[: frame + 1, :3])
        self.set_line(self.missile_marker, missile_position.reshape(1, 3))
        self.set_line(self.target_marker, target_position.reshape(1, 3))
        self.set_line(
            self.relative_line,
            np.vstack((missile_position, target_position)),
        )

        prediction = self.prediction_for_frame(frame)
        if prediction is None:
            self.prediction_path.set_data([], [])
            self.prediction_path.set_3d_properties([])
        else:
            self.set_line(self.prediction_path, prediction)

        time = frame * self.dt
        if frame == len(self.missile) - 1 and distance <= self.intercept_radius:
            result = "INTERCEPT"
            color = "#16825D"
        elif frame == len(self.missile) - 1:
            result = "FINISHED"
            color = "#B45309"
        else:
            result = "RUNNING"
            color = "#243B53"

        self.status_text.set_text(
            f"Time: {time:5.2f} s\n"
            f"Distance: {distance:7.2f} m\n"
            f"Status: {result}"
        )
        self.status_text.set_color(color)

        return (
            self.missile_path,
            self.target_path,
            self.prediction_path,
            self.relative_line,
            self.missile_marker,
            self.target_marker,
            self.status_text,
        )

    def create(self):
        interval_ms = 1000.0 * self.dt * self.frame_stride / self.playback_speed
        self.animation = FuncAnimation(
            self.figure,
            self.update,
            frames=self.frames,
            interval=interval_ms,
            blit=False,
            repeat=False,
        )
        return self.animation

    def save(self, output_path, fps=None, dpi=150):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.animation is None:
            self.create()

        if fps is None:
            fps = self.playback_speed / (self.dt * self.frame_stride)
        fps = max(float(fps), 1.0)

        if output_path.suffix.lower() == ".gif":
            writer = PillowWriter(fps=fps)
        elif output_path.suffix.lower() == ".mp4":
            writer = FFMpegWriter(fps=fps, bitrate=2400)
        else:
            raise ValueError("output_path must end with .gif or .mp4")

        self.animation.save(output_path, writer=writer, dpi=dpi)
        return output_path

    def show(self):
        if self.animation is None:
            self.create()
        plt.show()


def animate_interception(
    missile_trajectory,
    target_trajectory,
    predictions=None,
    prediction_origins=None,
    dt=0.05,
    playback_speed=1.0,
    intercept_radius=3.0,
    frame_stride=1,
    save_path=None,
):
    viewer = InterceptionAnimation(
        missile_trajectory=missile_trajectory,
        target_trajectory=target_trajectory,
        predictions=predictions,
        prediction_origins=prediction_origins,
        dt=dt,
        playback_speed=playback_speed,
        intercept_radius=intercept_radius,
        frame_stride=frame_stride,
    )
    viewer.create()
    if save_path is not None:
        viewer.save(save_path)
    viewer.show()
    return viewer


class ComparisonAnimation:
    """Play the two guidance methods side by side on one time axis."""

    def __init__(
        self,
        result,
        playback_speed=1.0,
        frame_stride=1,
    ):
        self.result = result
        self.target = np.asarray(
            result["target_trajectory"],
            dtype=float,
        )
        self.dt = float(result["dt"])
        self.intercept_radius = float(
            result["intercept_radius"]
        )
        self.playback_speed = float(playback_speed)
        self.frame_stride = int(frame_stride)

        self.methods = [
            (
                "acceleration",
                "A. Acceleration disturbance MPC",
                "LSTM 0.4 s position rollout",
            ),
            (
                "position",
                "B. Position interception MPC",
                "LSTM 5.0 s position prediction",
            ),
        ]

        self.validate_inputs()

        self.frames = list(
            range(
                0,
                len(self.target),
                self.frame_stride,
            )
        )

        if self.frames[-1] != len(self.target) - 1:
            self.frames.append(len(self.target) - 1)

        self.figure = plt.figure(figsize=(16, 8))
        self.axes = [
            self.figure.add_subplot(121, projection="3d"),
            self.figure.add_subplot(122, projection="3d"),
        ]

        self.artists = []
        self.configure_axes()
        self.create_artists()
        self.animation = None

    def validate_inputs(self):
        if self.target.ndim != 2 or self.target.shape[1] < 3:
            raise ValueError(
                "target_trajectory must have shape (steps, 3 or more)."
            )

        if len(self.target) == 0:
            raise ValueError("target_trajectory must not be empty.")

        if self.dt <= 0.0:
            raise ValueError("dt must be greater than zero.")

        if self.playback_speed <= 0.0:
            raise ValueError(
                "playback_speed must be greater than zero."
            )

        if self.frame_stride < 1:
            raise ValueError(
                "frame_stride must be at least one."
            )

        for key, _, _ in self.methods:
            missile = np.asarray(
                self.result[key]["missile_trajectory"],
                dtype=float,
            )

            if len(missile) != len(self.target):
                raise ValueError(
                    f"{key} trajectory must match target length."
                )

    @staticmethod
    def padded_limits(values, padding_ratio=0.08):
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        span = maximum - minimum
        padding = max(span * padding_ratio, 10.0)
        return minimum - padding, maximum + padding

    def configure_axes(self):
        positions = [self.target[:, :3]]

        for key, _, _ in self.methods:
            positions.append(
                np.asarray(
                    self.result[key]["missile_trajectory"],
                    dtype=float,
                )[:, :3]
            )

        positions = np.vstack(positions)
        x_limits = self.padded_limits(positions[:, 0])
        y_limits = self.padded_limits(positions[:, 1])
        z_limits = self.padded_limits(positions[:, 2])
        box_aspect = np.maximum(
            [
                x_limits[1] - x_limits[0],
                y_limits[1] - y_limits[0],
                z_limits[1] - z_limits[0],
            ],
            1.0,
        )

        for axis, (_, title, _) in zip(
            self.axes,
            self.methods,
        ):
            axis.set_xlim(x_limits)
            axis.set_ylim(y_limits)
            axis.set_zlim(z_limits)
            axis.set_box_aspect(box_aspect)
            axis.set_xlabel("X [m]")
            axis.set_ylabel("Y [m]")
            axis.set_zlabel("Z [m]")
            axis.set_title(title)
            axis.grid(alpha=0.25)
            axis.view_init(elev=24.0, azim=-58.0)

        self.figure.suptitle(
            "3D guidance comparison on the same target maneuver",
            fontsize=15,
            y=0.98,
        )

    def create_artists(self):
        for axis, (key, _, prediction_label) in zip(
            self.axes,
            self.methods,
        ):
            (missile_path,) = axis.plot(
                [],
                [],
                [],
                color="#FF7043",
                linewidth=2.2,
                label="Pursuer",
            )
            (target_path,) = axis.plot(
                [],
                [],
                [],
                color="#26C6B8",
                linewidth=2.4,
                label="Target",
            )
            (prediction_path,) = axis.plot(
                [],
                [],
                [],
                color="#263238",
                linestyle="--",
                linewidth=1.4,
                label=prediction_label,
            )
            (relative_line,) = axis.plot(
                [],
                [],
                [],
                color="#7E8A97",
                linestyle=":" if key == "acceleration" else "-",
                linewidth=1.4,
                alpha=0.8 if key == "acceleration" else 0.0,
                label="Current line of sight" if key == "acceleration" else None,
            )
            (hit_marker,) = axis.plot(
                [],
                [],
                [],
                marker="*",
                linestyle="None",
                color="#F2A900",
                markeredgecolor="#6B4E00",
                markersize=12,
                label=(
                    "Predicted intercept point"
                    if key == "position"
                    else None
                ),
            )
            (missile_marker,) = axis.plot(
                [],
                [],
                [],
                marker="o",
                color="#FF7043",
                markersize=7,
            )
            (target_marker,) = axis.plot(
                [],
                [],
                [],
                marker="o",
                color="#26C6B8",
                markersize=7,
            )
            status_text = axis.text2D(
                0.02,
                0.96,
                "",
                transform=axis.transAxes,
                va="top",
                fontsize=10,
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "white",
                    "edgecolor": "#C8D0D8",
                    "alpha": 0.92,
                },
            )

            axis.legend(loc="upper right", fontsize=8)
            self.artists.append(
                {
                    "missile_path": missile_path,
                    "target_path": target_path,
                    "prediction_path": prediction_path,
                    "relative_line": relative_line,
                    "hit_marker": hit_marker,
                    "missile_marker": missile_marker,
                    "target_marker": target_marker,
                    "status_text": status_text,
                }
            )

        self.figure.tight_layout(
            rect=(0.0, 0.0, 1.0, 0.92)
        )

    @staticmethod
    def set_line(line, points):
        points = np.asarray(points, dtype=float)
        line.set_data(points[:, 0], points[:, 1])
        line.set_3d_properties(points[:, 2])

    def prediction_for_frame(self, method, frame):
        predictions = method["predictions"]

        if frame >= len(predictions):
            return None

        prediction = predictions[frame]

        if prediction is None:
            return None

        prediction = np.asarray(prediction, dtype=float)

        if prediction.ndim != 2 or prediction.shape[1] < 3:
            return None

        origins = method["prediction_origins"]

        if frame < len(origins):
            origin = np.asarray(
                origins[frame],
                dtype=float,
            )[:3]
        else:
            origin = self.target[frame, :3]

        return np.vstack((origin, prediction[:, :3]))

    def display_time(self, frame, method):
        intercept_time = method["intercept_time"]
        time = frame * self.dt

        if not method["intercepted"] or intercept_time is None:
            return time

        return min(time, float(intercept_time))

    def interpolated_state(self, trajectory, time):
        position = time / self.dt
        lower = min(int(np.floor(position)), len(trajectory) - 1)
        upper = min(lower + 1, len(trajectory) - 1)
        fraction = float(np.clip(position - lower, 0.0, 1.0))

        state = (
            (1.0 - fraction) * trajectory[lower]
            + fraction * trajectory[upper]
        )

        path = trajectory[: lower + 1, :3]

        if fraction > 1e-9:
            path = np.vstack((path, state[:3]))

        return state, path, lower

    @staticmethod
    def velocity_from_state(state):
        speed = float(state[3])
        gamma = float(state[4])
        heading = float(state[5])

        return speed * np.array(
            [
                np.cos(gamma) * np.cos(heading),
                np.cos(gamma) * np.sin(heading),
                np.sin(gamma),
            ]
        )

    def cartesian_velocity(self, trajectory, time):
        position = time / self.dt
        lower = min(int(np.floor(position)), len(trajectory) - 1)
        upper = min(lower + 1, len(trajectory) - 1)

        if upper == lower:
            return self.velocity_from_state(trajectory[lower])

        return (
            trajectory[upper, :3] - trajectory[lower, :3]
        ) / self.dt

    @staticmethod
    def line_of_sight_metrics(
        target_position,
        missile_position,
        target_velocity,
        missile_velocity,
    ):
        relative_position = target_position - missile_position
        distance = max(
            float(np.linalg.norm(relative_position)),
            1e-9,
        )
        line_of_sight = relative_position / distance
        relative_velocity = target_velocity - missile_velocity
        closing_speed = -float(
            np.dot(relative_velocity, line_of_sight)
        )
        los_rate = (
            np.linalg.norm(
                np.cross(relative_position, relative_velocity)
            )
            / distance**2
        )

        return float(np.degrees(los_rate)), closing_speed

    @staticmethod
    def item_for_frame(items, frame):
        if items is None or frame >= len(items):
            return None

        return items[frame]

    def update_method(self, key, frame, method, artists):
        missile = np.asarray(
            method["missile_trajectory"],
            dtype=float,
        )
        time = self.display_time(frame, method)
        target_state, target_path, sample_frame = (
            self.interpolated_state(self.target, time)
        )
        missile_state, missile_path, _ = self.interpolated_state(
            missile,
            time,
        )
        missile_position = missile_state[:3]
        target_position = target_state[:3]
        distance = float(
            np.linalg.norm(target_position - missile_position)
        )
        relative_path = target_path - missile_path
        distances_so_far = np.linalg.norm(relative_path, axis=1)
        closest_so_far = float(np.min(distances_so_far))

        self.set_line(
            artists["missile_path"],
            missile_path,
        )
        self.set_line(
            artists["target_path"],
            target_path,
        )
        self.set_line(
            artists["missile_marker"],
            missile_position.reshape(1, 3),
        )
        self.set_line(
            artists["target_marker"],
            target_position.reshape(1, 3),
        )
        if key == "acceleration":
            self.set_line(
                artists["relative_line"],
                np.vstack((missile_position, target_position)),
            )
        else:
            artists["relative_line"].set_data([], [])
            artists["relative_line"].set_3d_properties([])

        prediction = self.prediction_for_frame(
            method,
            sample_frame,
        )

        if prediction is None:
            artists["prediction_path"].set_data([], [])
            artists["prediction_path"].set_3d_properties([])
        else:
            self.set_line(
                artists["prediction_path"],
                prediction,
            )

        hit_point = None

        if key == "position":
            hit_point = self.item_for_frame(
                method.get("hit_points"),
                sample_frame,
            )

        if hit_point is None:
            artists["hit_marker"].set_data([], [])
            artists["hit_marker"].set_3d_properties([])
        else:
            self.set_line(
                artists["hit_marker"],
                np.asarray(hit_point).reshape(1, 3),
            )

        intercept_time = method["intercept_time"]

        if (
            method["intercepted"]
            and intercept_time is not None
            and time >= intercept_time
        ):
            status = "INTERCEPT"
            color = "#16825D"
        elif frame == len(self.target) - 1:
            status = "FINISHED"
            color = "#B45309"
        else:
            status = "RUNNING"
            color = "#243B53"

        time_text = (
            "--"
            if intercept_time is None or time < intercept_time
            else f"{intercept_time:.2f} s"
        )

        lines = [
            f"Time: {time:5.2f} s",
            f"Distance: {distance:7.2f} m",
            f"Closest so far: {closest_so_far:7.2f} m",
        ]

        if key == "acceleration":
            target_velocity = self.cartesian_velocity(
                self.target,
                time,
            )
            missile_velocity = self.cartesian_velocity(
                missile,
                time,
            )
            los_rate, closing_speed = self.line_of_sight_metrics(
                target_position,
                missile_position,
                target_velocity,
                missile_velocity,
            )
            los_rate_text = (
                "converged"
                if distance <= 2.0 * self.intercept_radius
                else f"{los_rate:6.2f} deg/s"
            )
            lines.extend(
                [
                    f"LOS rate: {los_rate_text}",
                    f"Closing speed: {closing_speed:7.2f} m/s",
                ]
            )
        else:
            hit_step = self.item_for_frame(
                method.get("hit_steps"),
                sample_frame,
            )
            hit_time = self.item_for_frame(
                method.get("hit_times"),
                sample_frame,
            )
            reachable = self.item_for_frame(
                method.get("hit_reachable"),
                sample_frame,
            )

            if hit_step is not None and hit_time is not None:
                reachability = (
                    "reachable" if reachable else "best candidate"
                )
                lines.extend(
                    [
                        f"Selected j_hit: {hit_step}",
                        f"Predicted hit in: {hit_time:.2f} s",
                        f"Candidate: {reachability}",
                    ]
                )

        lines.extend(
            [
                f"Intercept time: {time_text}",
                f"Status: {status}",
            ]
        )
        artists["status_text"].set_text("\n".join(lines))
        artists["status_text"].set_color(color)

    def update(self, frame):
        for (key, _, _), artists in zip(
            self.methods,
            self.artists,
        ):
            self.update_method(
                key,
                frame,
                self.result[key],
                artists,
            )

        return tuple(
            artist
            for group in self.artists
            for artist in group.values()
        )

    def create(self):
        interval_ms = (
            1000.0
            * self.dt
            * self.frame_stride
            / self.playback_speed
        )

        self.animation = FuncAnimation(
            self.figure,
            self.update,
            frames=self.frames,
            interval=interval_ms,
            blit=False,
            repeat=False,
        )

        return self.animation

    def save(self, output_path, fps=None, dpi=150):
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.animation is None:
            self.create()

        if fps is None:
            fps = (
                self.playback_speed
                / (self.dt * self.frame_stride)
            )

        fps = max(float(fps), 1.0)

        if output_path.suffix.lower() == ".gif":
            writer = PillowWriter(fps=fps)
        elif output_path.suffix.lower() == ".mp4":
            writer = FFMpegWriter(
                fps=fps,
                bitrate=2400,
            )
        else:
            raise ValueError(
                "output_path must end with .gif or .mp4"
            )

        self.animation.save(
            output_path,
            writer=writer,
            dpi=dpi,
        )

        return output_path

    def show(self):
        if self.animation is None:
            self.create()

        plt.show()


def animate_comparison(
    result,
    playback_speed=1.0,
    frame_stride=1,
    save_path=None,
):
    viewer = ComparisonAnimation(
        result=result,
        playback_speed=playback_speed,
        frame_stride=frame_stride,
    )
    viewer.create()

    if save_path is not None:
        viewer.save(save_path)

    viewer.show()
    return viewer
