import numpy as np


class ProportionalNavigation:
    """Three-dimensional true proportional-navigation guidance.

    Both states use the project convention
    ``[x, y, z, speed, flight_path_angle, heading]``.  ``solve`` returns
    ``[vertical_acceleration, side_acceleration]`` for ``Missile``.
    """

    def __init__(
        self,
        navigation_constant=3.0,
        max_g=20.0,
    ):
        if navigation_constant <= 0.0:
            raise ValueError("navigation_constant must be greater than zero.")
        if max_g <= 0.0:
            raise ValueError("max_g must be greater than zero.")

        self.N = float(navigation_constant)
        self.max_acceleration = float(max_g) * 9.81

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
    def local_basis(velocity):
        """Return missile-fixed forward, vertical, and side unit vectors."""
        velocity = np.asarray(velocity, dtype=float)
        speed = np.linalg.norm(velocity)

        if speed < 1e-8:
            raise ValueError("velocity magnitude must be greater than zero.")

        forward = velocity / speed
        world_up = np.array([0.0, 0.0, 1.0])

        side = np.cross(world_up, forward)

        # world_up and forward are parallel in near-vertical flight, so use a
        # different reference axis to keep a valid local frame.
        if np.linalg.norm(side) < 1e-8:
            side = np.cross(
                np.array([0.0, 1.0, 0.0]),
                forward,
            )

        side = side / np.linalg.norm(side)
        vertical = np.cross(forward, side)
        vertical = vertical / np.linalg.norm(vertical)

        return forward, vertical, side

    def solve(self, missile_state, target_state):
        """Return the PN command ``[vertical, side]`` in m/s^2."""
        missile_state = np.asarray(missile_state, dtype=float)
        target_state = np.asarray(target_state, dtype=float)

        if missile_state.shape != (6,) or target_state.shape != (6,):
            raise ValueError(
                "states must be [x, y, z, speed, flight_path_angle, heading]."
            )

        missile_position = missile_state[0:3]
        target_position = target_state[0:3]

        missile_velocity = missile_state[3] * self.direction(
            missile_state[4],
            missile_state[5],
        )
        target_velocity = target_state[3] * self.direction(
            target_state[4],
            target_state[5],
        )

        relative_position = target_position - missile_position
        distance = np.linalg.norm(relative_position)

        if distance < 1e-8:
            return np.zeros(2)

        los_direction = relative_position / distance
        relative_velocity = target_velocity - missile_velocity
        closing_speed = -np.dot(relative_velocity, los_direction)

        # Classical PN assumes a closing engagement.  Acquisition guidance for
        # an opening target is intentionally outside this baseline controller.
        if closing_speed <= 0.0:
            return np.zeros(2)

        los_angular_velocity = (
            np.cross(relative_position, relative_velocity) / distance**2
        )

        acceleration_world = (
            self.N
            * closing_speed
            * np.cross(los_angular_velocity, los_direction)
        )

        _, vertical, side = self.local_basis(missile_velocity)
        command = np.array(
            [
                np.dot(acceleration_world, vertical),
                np.dot(acceleration_world, side),
            ]
        )

        command_norm = np.linalg.norm(command)

        if command_norm > self.max_acceleration:
            command = command / command_norm * self.max_acceleration

        return command
