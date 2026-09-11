import numpy as np
from qpsolvers import solve_qp
from mpc_position import PositionInterceptionMPC


class AccelerationDisturbanceMPC:
    def __init__(
        self,
        dt=0.05,
        horizon=20,
        disturbance_steps=8,
        max_g=20.0,
        los_velocity_scale=20.0,
        terminal_range_weight=20.0,
        control_weight=0.02,
        smooth_weight=0.05,
    ):
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.disturbance_steps = int(disturbance_steps)

        self.max_g = float(max_g)
        self.u_max = self.max_g * 9.81

        self.los_velocity_scale = float(los_velocity_scale)
        self.terminal_range_weight = float(terminal_range_weight)
        self.control_weight = float(control_weight)
        self.smooth_weight = float(smooth_weight)

        self.U = None
        self.last_u = np.zeros(2)
        self.solver_failures = 0

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
    def safe_cos(angle):
        cos_angle = np.cos(angle)

        if abs(cos_angle) < 1e-3:
            return np.copysign(1e-3, cos_angle if cos_angle != 0.0 else 1.0)

        return cos_angle

    def model(
        self,
        state,
        control,
        pursuer_speed,
        disturbance,
    ):
        state = np.asarray(state, dtype=float)
        control = np.asarray(control, dtype=float)
        disturbance = np.asarray(disturbance, dtype=float)

        relative_position = state[0:3]

        target_speed = max(state[3], 1e-6)
        target_gamma = state[4]
        target_heading = state[5]

        pursuer_gamma = state[6]
        pursuer_heading = state[7]

        target_parallel = disturbance[0]
        target_vertical = disturbance[1]
        target_side = disturbance[2]

        pursuer_vertical = control[0]
        pursuer_side = control[1]

        next_target_speed = max(
            target_speed + target_parallel * self.dt,
            1e-6,
        )

        next_target_gamma = (
            target_gamma
            + target_vertical / target_speed * self.dt
        )

        next_target_heading = (
            target_heading
            + target_side
            / (target_speed * self.safe_cos(target_gamma))
            * self.dt
        )

        safe_pursuer_speed = max(float(pursuer_speed), 1e-6)

        next_pursuer_gamma = (
            pursuer_gamma
            + pursuer_vertical / safe_pursuer_speed * self.dt
        )

        next_pursuer_heading = (
            pursuer_heading
            + pursuer_side
            / (
                safe_pursuer_speed
                * self.safe_cos(pursuer_gamma)
            )
            * self.dt
        )

        target_velocity = (
            next_target_speed
            * self.direction(
                next_target_gamma,
                next_target_heading,
            )
        )

        pursuer_velocity = (
            safe_pursuer_speed
            * self.direction(
                next_pursuer_gamma,
                next_pursuer_heading,
            )
        )

        next_relative_position = (
            relative_position
            + (target_velocity - pursuer_velocity) * self.dt
        )

        return np.array(
            [
                next_relative_position[0],
                next_relative_position[1],
                next_relative_position[2],
                next_target_speed,
                next_target_gamma,
                next_target_heading,
                next_pursuer_gamma,
                next_pursuer_heading,
            ]
        )

    def linearize(
        self,
        state,
        control,
        pursuer_speed,
        disturbance,
    ):
        state = np.asarray(state, dtype=float)
        control = np.asarray(control, dtype=float)
        disturbance = np.asarray(disturbance, dtype=float)

        A = np.zeros((8, 8))
        B = np.zeros((8, 2))
        E = np.zeros((8, 3))

        dt = self.dt
        target_speed_state = state[3]
        target_speed = max(target_speed_state, 1e-6)
        target_gamma = state[4]
        target_heading = state[5]
        pursuer_gamma = state[6]
        pursuer_heading = state[7]

        target_parallel = disturbance[0]
        target_vertical = disturbance[1]
        target_side = disturbance[2]
        pursuer_vertical = control[0]
        pursuer_side = control[1]

        pursuer_speed = max(float(pursuer_speed), 1e-6)
        target_cos = self.safe_cos(target_gamma)
        pursuer_cos = self.safe_cos(pursuer_gamma)

        target_cos_derivative = (
            np.sin(target_gamma) / target_cos**2
            if abs(np.cos(target_gamma)) >= 1e-3
            else 0.0
        )
        pursuer_cos_derivative = (
            np.sin(pursuer_gamma) / pursuer_cos**2
            if abs(np.cos(pursuer_gamma)) >= 1e-3
            else 0.0
        )

        current_speed_active = float(target_speed_state > 1e-6)
        raw_next_target_speed = target_speed + target_parallel * dt
        next_speed_active = float(raw_next_target_speed > 1e-6)
        next_target_speed = max(raw_next_target_speed, 1e-6)

        next_target_gamma = (
            target_gamma + target_vertical / target_speed * dt
        )
        next_target_heading = (
            target_heading + target_side / (target_speed * target_cos) * dt
        )
        next_pursuer_gamma = (
            pursuer_gamma + pursuer_vertical / pursuer_speed * dt
        )
        next_pursuer_heading = (
            pursuer_heading + pursuer_side / (pursuer_speed * pursuer_cos) * dt
        )

        target_direction = self.direction(
            next_target_gamma,
            next_target_heading,
        )
        target_gamma_jacobian = np.array(
            [
                -np.sin(next_target_gamma) * np.cos(next_target_heading),
                -np.sin(next_target_gamma) * np.sin(next_target_heading),
                np.cos(next_target_gamma),
            ]
        )
        target_heading_jacobian = np.array(
            [
                -np.cos(next_target_gamma) * np.sin(next_target_heading),
                np.cos(next_target_gamma) * np.cos(next_target_heading),
                0.0,
            ]
        )
        pursuer_gamma_jacobian = np.array(
            [
                -np.sin(next_pursuer_gamma) * np.cos(next_pursuer_heading),
                -np.sin(next_pursuer_gamma) * np.sin(next_pursuer_heading),
                np.cos(next_pursuer_gamma),
            ]
        )
        pursuer_heading_jacobian = np.array(
            [
                -np.cos(next_pursuer_gamma) * np.sin(next_pursuer_heading),
                np.cos(next_pursuer_gamma) * np.cos(next_pursuer_heading),
                0.0,
            ]
        )

        d_speed_next_d_speed = next_speed_active * current_speed_active
        d_target_gamma_d_speed = (
            -target_vertical * dt / target_speed**2 * current_speed_active
        )
        d_target_heading_d_speed = (
            -target_side
            * dt
            / (target_speed**2 * target_cos)
            * current_speed_active
        )
        d_target_heading_d_gamma = (
            target_side / target_speed * dt * target_cos_derivative
        )
        d_pursuer_heading_d_gamma = (
            pursuer_side / pursuer_speed * dt * pursuer_cos_derivative
        )

        d_target_velocity_d_speed = (
            d_speed_next_d_speed * target_direction
            + next_target_speed
            * (
                target_gamma_jacobian * d_target_gamma_d_speed
                + target_heading_jacobian * d_target_heading_d_speed
            )
        )
        d_target_velocity_d_gamma = next_target_speed * (
            target_gamma_jacobian
            + target_heading_jacobian * d_target_heading_d_gamma
        )
        d_target_velocity_d_heading = (
            next_target_speed * target_heading_jacobian
        )
        d_pursuer_velocity_d_gamma = pursuer_speed * (
            pursuer_gamma_jacobian
            + pursuer_heading_jacobian * d_pursuer_heading_d_gamma
        )
        d_pursuer_velocity_d_heading = (
            pursuer_speed * pursuer_heading_jacobian
        )

        A[0:3, 0:3] = np.eye(3)
        A[0:3, 3] = dt * d_target_velocity_d_speed
        A[0:3, 4] = dt * d_target_velocity_d_gamma
        A[0:3, 5] = dt * d_target_velocity_d_heading
        A[0:3, 6] = -dt * d_pursuer_velocity_d_gamma
        A[0:3, 7] = -dt * d_pursuer_velocity_d_heading
        A[3, 3] = d_speed_next_d_speed
        A[4, 3] = d_target_gamma_d_speed
        A[4, 4] = 1.0
        A[5, 3] = d_target_heading_d_speed
        A[5, 4] = d_target_heading_d_gamma
        A[5, 5] = 1.0
        A[6, 6] = 1.0
        A[7, 6] = d_pursuer_heading_d_gamma
        A[7, 7] = 1.0

        B[0:3, 0] = -dt**2 * pursuer_gamma_jacobian
        B[0:3, 1] = -dt**2 / pursuer_cos * pursuer_heading_jacobian
        B[6, 0] = dt / pursuer_speed
        B[7, 1] = dt / (pursuer_speed * pursuer_cos)

        E[0:3, 0] = dt**2 * next_speed_active * target_direction
        E[0:3, 1] = (
            dt**2
            * next_target_speed
            / target_speed
            * target_gamma_jacobian
        )
        E[0:3, 2] = (
            dt**2
            * next_target_speed
            / (target_speed * target_cos)
            * target_heading_jacobian
        )
        E[3, 0] = dt * next_speed_active
        E[4, 1] = dt / target_speed
        E[5, 2] = dt / (target_speed * target_cos)

        next_state = self.model(
            state,
            control,
            pursuer_speed,
            disturbance,
        )

        affine = (
            next_state
            - A @ state
            - B @ control
            - E @ disturbance
        )

        return A, B, E, affine

    def rollout(
        self,
        initial_state,
        controls,
        pursuer_speed,
        disturbances,
    ):
        states = [np.asarray(initial_state, dtype=float)]

        for step in range(self.horizon):
            next_state = self.model(
                states[-1],
                controls[step],
                pursuer_speed,
                disturbances[step],
            )

            states.append(next_state)

        return np.asarray(states)

    def prediction_matrices(
        self,
        nominal_states,
        controls,
        pursuer_speed,
        disturbances,
    ):
        S_k = np.zeros((8, 2 * self.horizon))
        T_k = np.eye(8)
        h_k = np.zeros(8)

        S_blocks = []
        T_blocks = []
        h_blocks = []

        for step in range(self.horizon):
            A, B, E, affine = self.linearize(
                nominal_states[step],
                controls[step],
                pursuer_speed,
                disturbances[step],
            )

            S_k = A @ S_k
            S_k[:, 2 * step : 2 * (step + 1)] += B

            T_k = A @ T_k
            h_k = A @ h_k + E @ disturbances[step] + affine

            S_blocks.append(S_k.copy())
            T_blocks.append(T_k.copy())
            h_blocks.append(h_k.copy())

        S = np.vstack(S_blocks)
        T = np.vstack(T_blocks)
        h = np.concatenate(h_blocks)

        return S, T, h

    def los_velocity(self, state, pursuer_speed):
        relative_position = state[0:3]
        distance = max(np.linalg.norm(relative_position), 1e-6)
        los_direction = relative_position / distance

        target_velocity = (
            state[3]
            * self.direction(state[4], state[5])
        )

        pursuer_velocity = (
            pursuer_speed
            * self.direction(state[6], state[7])
        )

        relative_velocity = target_velocity - pursuer_velocity

        return (
            relative_velocity
            - np.dot(relative_velocity, los_direction) * los_direction
        )

    def los_velocity_linearization(self, state, pursuer_speed):
        state = np.asarray(state, dtype=float)
        relative_position = state[0:3]
        target_speed = state[3]
        target_gamma = state[4]
        target_heading = state[5]
        pursuer_gamma = state[6]
        pursuer_heading = state[7]
        pursuer_speed = max(float(pursuer_speed), 1e-6)

        raw_distance = np.linalg.norm(relative_position)
        distance = max(raw_distance, 1e-6)
        los_direction = relative_position / distance
        projection = np.eye(3) - np.outer(los_direction, los_direction)

        if raw_distance > 1e-6:
            los_jacobian = projection / distance
        else:
            los_jacobian = np.eye(3) / distance

        target_direction = self.direction(target_gamma, target_heading)
        pursuer_direction = self.direction(pursuer_gamma, pursuer_heading)

        target_gamma_jacobian = np.array(
            [
                -np.sin(target_gamma) * np.cos(target_heading),
                -np.sin(target_gamma) * np.sin(target_heading),
                np.cos(target_gamma),
            ]
        )
        target_heading_jacobian = np.array(
            [
                -np.cos(target_gamma) * np.sin(target_heading),
                np.cos(target_gamma) * np.cos(target_heading),
                0.0,
            ]
        )
        pursuer_gamma_jacobian = np.array(
            [
                -np.sin(pursuer_gamma) * np.cos(pursuer_heading),
                -np.sin(pursuer_gamma) * np.sin(pursuer_heading),
                np.cos(pursuer_gamma),
            ]
        )
        pursuer_heading_jacobian = np.array(
            [
                -np.cos(pursuer_gamma) * np.sin(pursuer_heading),
                np.cos(pursuer_gamma) * np.cos(pursuer_heading),
                0.0,
            ]
        )

        relative_velocity = (
            target_speed * target_direction
            - pursuer_speed * pursuer_direction
        )
        radial_velocity = np.dot(relative_velocity, los_direction)

        gradient = np.zeros((3, 8))
        gradient[:, 0:3] = -(
            radial_velocity * np.eye(3)
            + np.outer(los_direction, relative_velocity)
        ) @ los_jacobian
        gradient[:, 3] = projection @ target_direction
        gradient[:, 4] = (
            projection @ (target_speed * target_gamma_jacobian)
        )
        gradient[:, 5] = (
            projection @ (target_speed * target_heading_jacobian)
        )
        gradient[:, 6] = (
            projection @ (-pursuer_speed * pursuer_gamma_jacobian)
        )
        gradient[:, 7] = (
            projection @ (-pursuer_speed * pursuer_heading_jacobian)
        )

        los_velocity = projection @ relative_velocity
        offset = los_velocity - gradient @ state

        return gradient, offset

    def disturbance_schedule(self, predicted_accelerations):
        predicted_accelerations = np.asarray(
            predicted_accelerations,
            dtype=float,
        )

        if (
            predicted_accelerations.ndim != 2
            or predicted_accelerations.shape[1] != 3
        ):
            raise ValueError(
                "predicted_accelerations must have shape (steps, 3)."
            )

        disturbances = np.zeros((self.horizon, 3))

        used_steps = min(
            self.disturbance_steps,
            len(predicted_accelerations),
            self.horizon,
        )

        disturbances[:used_steps] = predicted_accelerations[:used_steps]

        return disturbances

    def solve(
        self,
        pursuer_state,
        target_state,
        predicted_accelerations,
    ):
        pursuer_state = np.asarray(pursuer_state, dtype=float)
        target_state = np.asarray(target_state, dtype=float)

        disturbances = self.disturbance_schedule(
            predicted_accelerations
        )

        pursuer_speed = max(pursuer_state[3], 1e-6)

        initial_state = np.array(
            [
                target_state[0] - pursuer_state[0],
                target_state[1] - pursuer_state[1],
                target_state[2] - pursuer_state[2],
                target_state[3],
                target_state[4],
                target_state[5],
                pursuer_state[4],
                pursuer_state[5],
            ]
        )

        if self.U is None or self.U.shape != (self.horizon, 2):
            controls = np.zeros((self.horizon, 2))
        else:
            controls = np.vstack(
                (
                    self.U[1:],
                    self.U[-1],
                )
            )

        nominal_states = self.rollout(
            initial_state,
            controls,
            pursuer_speed,
            disturbances,
        )

        S, T, h = self.prediction_matrices(
            nominal_states,
            controls,
            pursuer_speed,
            disturbances,
        )

        free_states = T @ initial_state + h

        los_rows = np.zeros(
            (3 * self.horizon, 8 * self.horizon)
        )
        los_offsets = np.zeros(3 * self.horizon)

        for step in range(self.horizon):
            gradient, offset = self.los_velocity_linearization(
                nominal_states[step + 1],
                pursuer_speed,
            )

            row = slice(3 * step, 3 * (step + 1))
            column = slice(8 * step, 8 * (step + 1))

            los_rows[row, column] = gradient
            los_offsets[row] = offset

        los_control = los_rows @ S
        los_free = los_rows @ free_states + los_offsets

        los_weight = 1.0 / self.los_velocity_scale**2

        terminal_selector = np.zeros(
            (3, 8 * self.horizon)
        )
        terminal_selector[:, -8:-5] = np.eye(3)

        range_control = terminal_selector @ S
        range_free = terminal_selector @ free_states

        initial_range = max(
            np.linalg.norm(initial_state[0:3]),
            100.0,
        )
        range_weight = (
            self.terminal_range_weight / initial_range**2
        )

        variable_count = 2 * self.horizon
        difference = np.eye(variable_count)

        for step in range(1, self.horizon):
            current = slice(2 * step, 2 * (step + 1))
            previous = slice(2 * (step - 1), 2 * step)
            difference[current, previous] -= np.eye(2)

        difference_reference = np.zeros(variable_count)
        difference_reference[0:2] = self.last_u

        control_scale_squared = self.u_max**2

        hessian = (
            los_weight * los_control.T @ los_control
            + range_weight * range_control.T @ range_control
            + self.control_weight
            / control_scale_squared
            * np.eye(variable_count)
            + self.smooth_weight
            / control_scale_squared
            * difference.T
            @ difference
        )

        gradient = (
            los_weight * los_control.T @ los_free
            + range_weight * range_control.T @ range_free
            - self.smooth_weight
            / control_scale_squared
            * difference.T
            @ difference_reference
        )

        P = 2.0 * hessian
        q = 2.0 * gradient

        P = (
            0.5 * (P + P.T)
            + 1e-9 * np.eye(variable_count)
        )

        axis_limit = self.u_max / np.sqrt(2.0)

        solution = solve_qp(
            P,
            q,
            lb=np.full(variable_count, -axis_limit),
            ub=np.full(variable_count, axis_limit),
            solver="quadprog",
        )

        if solution is None:
            self.solver_failures += 1
            self.U = controls
        else:
            self.U = solution.reshape(self.horizon, 2)

        self.last_u = self.U[0].copy()

        return self.last_u.copy()


MPC = AccelerationDisturbanceMPC


__all__ = [
    "AccelerationDisturbanceMPC",
    "PositionInterceptionMPC",
    "MPC",
]
