import numpy as np


class Missile:

    def __init__(
        self,
        x=0.0,
        y=0.0,
        z=0.0,
        v=400.0,
        flight_path_angle=0.0,
        heading=0.0,
        max_g=20.0,
    ):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

        self.v = float(v)
        self.gamma = float(flight_path_angle)
        self.head = float(heading)

        self.g = 9.81
        self.max_acceleration = float(max_g) * self.g
        self.max_flight_path_angle = np.radians(80.0)

        self.vertical_acceleration = 0.0
        self.side_acceleration = 0.0

        self.gamma_rate = 0.0
        self.head_rate = 0.0

    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

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

    def update_acceleration(self, acceleration):
        acceleration = np.asarray(acceleration, dtype=float)

        if acceleration.shape != (2,):
            raise ValueError(
                "acceleration must be [vertical, side]."
            )

        acceleration_norm = np.linalg.norm(acceleration)

        if acceleration_norm > self.max_acceleration:
            acceleration = (
                acceleration
                / acceleration_norm
                * self.max_acceleration
            )

        self.vertical_acceleration = acceleration[0]
        self.side_acceleration = acceleration[1]

    def get_velocity(self):
        return self.v * self.direction(self.gamma, self.head)

    def step(self, dt):
        dt = float(dt)

        if dt <= 0.0:
            raise ValueError("dt must be greater than zero.")

        safe_speed = max(self.v, 1e-6)
        horizontal_speed = safe_speed * np.cos(self.gamma)
        safe_horizontal_speed = max(abs(horizontal_speed), 1e-6)

        self.gamma_rate = self.vertical_acceleration / safe_speed
        self.head_rate = self.side_acceleration / safe_horizontal_speed

        self.gamma = np.clip(
            self.gamma + self.gamma_rate * dt,
            -self.max_flight_path_angle,
            self.max_flight_path_angle,
        )

        self.head = self.wrap_angle(
            self.head + self.head_rate * dt
        )

        velocity = self.get_velocity()

        self.x += velocity[0] * dt
        self.y += velocity[1] * dt
        self.z += velocity[2] * dt

    def get_state(self):
        return np.array(
            [
                self.x,
                self.y,
                self.z,
                self.v,
                self.gamma,
                self.head,
            ]
        )

    def get_pos_vel(self):
        return np.concatenate(
            (
                np.array([self.x, self.y, self.z]),
                self.get_velocity(),
            )
        )
