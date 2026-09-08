import numpy as np


class Vehicle:

    def __init__(
        self,
        x=0.0,
        y=0.0,
        z=0.0,
        v=100.0,
        flight_path_angle=0.0,
        heading=0.0,
    ):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

        self.v = float(v)
        self.gamma = float(flight_path_angle)
        self.head = float(heading)

        self.g = 9.81
        self.min_speed = 70.0
        self.max_speed = 220.0
        self.max_flight_path_angle = np.radians(80.0)

        self.max_parallel_acceleration = 0.5 * self.g
        self.max_normal_acceleration = 8.0 * self.g

        self.acceleration_cmd = np.zeros(3)
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

    def local_basis(self):
        cos_gamma = np.cos(self.gamma)
        sin_gamma = np.sin(self.gamma)
        cos_heading = np.cos(self.head)
        sin_heading = np.sin(self.head)

        forward = np.array(
            [
                cos_gamma * cos_heading,
                cos_gamma * sin_heading,
                sin_gamma,
            ]
        )

        vertical = np.array(
            [
                -sin_gamma * cos_heading,
                -sin_gamma * sin_heading,
                cos_gamma,
            ]
        )

        side = np.array(
            [
                -sin_heading,
                cos_heading,
                0.0,
            ]
        )

        return forward, vertical, side

    def set_acceleration_cmd(self, acceleration_cmd):
        acceleration_cmd = np.asarray(acceleration_cmd, dtype=float)

        self.acceleration_cmd = np.array(
            [
                np.clip(
                    acceleration_cmd[0],
                    -self.max_parallel_acceleration,
                    self.max_parallel_acceleration,
                ),
                np.clip(
                    acceleration_cmd[1],
                    -self.max_normal_acceleration,
                    self.max_normal_acceleration,
                ),
                np.clip(
                    acceleration_cmd[2],
                    -self.max_normal_acceleration,
                    self.max_normal_acceleration,
                ),
            ]
        )

    def get_velocity(self):
        return self.v * self.direction(self.gamma, self.head)

    def get_acceleration(self):
        forward, vertical, side = self.local_basis()

        return (
            self.acceleration_cmd[0] * forward
            + self.acceleration_cmd[1] * vertical
            + self.acceleration_cmd[2] * side
        )

    def step(self, dt):
        dt = float(dt)

        parallel_acceleration = self.acceleration_cmd[0]
        vertical_acceleration = self.acceleration_cmd[1]
        side_acceleration = self.acceleration_cmd[2]

        safe_speed = max(self.v, 1e-6)
        horizontal_speed = safe_speed * np.cos(self.gamma)
        safe_horizontal_speed = max(abs(horizontal_speed), 1e-6)

        self.gamma_rate = vertical_acceleration / safe_speed
        self.head_rate = side_acceleration / safe_horizontal_speed

        self.v = np.clip(
            self.v + parallel_acceleration * dt,
            self.min_speed,
            self.max_speed,
        )

        self.gamma = np.clip(
            self.gamma + self.gamma_rate * dt,
            -self.max_flight_path_angle,
            self.max_flight_path_angle,
        )

        self.head = self.wrap_angle(self.head + self.head_rate * dt)

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
