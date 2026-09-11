from dataclasses import dataclass

import numpy as np
from qpsolvers import solve_qp
from qpsolvers.exceptions import ProblemError
from scipy import sparse


@dataclass(frozen=True)
class InterceptionEstimate:
    j_hit: int
    t_hit: float
    target_position: np.ndarray
    minimum_miss: float
    reachable: bool
    fallback: str | None

    @property
    def target_index(self):
        return self.j_hit - 1


class PositionInterceptionMPC:

    def __init__(
        self,
        dt=0.05,
        max_g=20.0,
        intercept_radius=3.0,
        max_horizon=100,
        tracking_weight=0.02,
        hit_weight=200.0,
        control_weight=0.01,
        reachability_iterations=2,
        candidate_stride=10,
        estimate_update_interval=5,
        control_block_size=5,
        qp_solver="quadprog",
    ):
        self.dt = float(dt)
        self.max_g = float(max_g)
        self.u_max = self.max_g * 9.81
        self.intercept_radius = float(intercept_radius)
        self.max_horizon = int(max_horizon)

        self.tracking_weight = float(tracking_weight)
        self.hit_weight = float(hit_weight)
        self.control_weight = float(control_weight)
        self.reachability_iterations = int(
            reachability_iterations
        )
        self.candidate_stride = int(candidate_stride)
        self.estimate_update_interval = int(
            estimate_update_interval
        )
        self.control_block_size = int(control_block_size)
        self.qp_solver = str(qp_solver)

        self.max_flight_path_angle = np.radians(80.0)
        self.U = None
        self.last_info = None
        self.last_prediction = None
        self.solver_failures = 0
        self.qp_solve_count = 0
        self.solve_count = 0
        self.cost_cache = {}
        self.bound_cache = {}
        self.expansion_cache = {}

        if self.candidate_stride < 1:
            raise ValueError(
                "candidate_stride must be at least one."
            )
        if self.estimate_update_interval < 1:
            raise ValueError(
                "estimate_update_interval must be at least one."
            )
        if self.control_block_size < 1:
            raise ValueError(
                "control_block_size must be at least one."
            )

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

    @staticmethod
    def safe_cos(angle):
        cos_angle = np.cos(angle)

        if abs(cos_angle) < 1e-3:
            return np.copysign(
                1e-3,
                cos_angle if cos_angle != 0.0 else 1.0,
            )

        return cos_angle

    def limit_control(self, control):
        control = np.asarray(control, dtype=float)
        control_norm = np.linalg.norm(control)

        if control_norm > self.u_max:
            control = control / control_norm * self.u_max

        return control

    """
        state:
        [px, py, pz, pursuer_gamma, pursuer_heading]

        control:
        [pursuer_vertical_acceleration, pursuer_side_acceleration]
    """

    def model(self, state, control, speed):
        state = np.asarray(state, dtype=float)
        control = self.limit_control(control)

        position = state[0:3]
        flight_path_angle = state[3]
        heading = state[4]

        safe_speed = max(float(speed), 1e-6)
        vertical_acceleration = control[0]
        side_acceleration = control[1]

        next_flight_path_angle = np.clip(
            flight_path_angle
            + vertical_acceleration / safe_speed * self.dt,
            -self.max_flight_path_angle,
            self.max_flight_path_angle,
        )

        next_heading = self.wrap_angle(
            heading
            + side_acceleration
            / (
                safe_speed
                * self.safe_cos(flight_path_angle)
            )
            * self.dt
        )

        velocity = safe_speed * self.direction(
            next_flight_path_angle,
            next_heading,
        )

        next_position = position + velocity * self.dt

        return np.array(
            [
                next_position[0],
                next_position[1],
                next_position[2],
                next_flight_path_angle,
                next_heading,
            ]
        )

    def linearize(self, state, control, speed):
        state = np.asarray(state, dtype=float)
        control = np.asarray(control, dtype=float)
        control = self.limit_control(control)

        dt = self.dt
        speed = max(float(speed), 1e-6)
        gamma = float(state[3])
        heading = float(state[4])
        vertical_acceleration = float(control[0])
        side_acceleration = float(control[1])

        cosine = np.cos(gamma)
        safe_cosine = self.safe_cos(gamma)
        safe_cosine_derivative = (
            np.sin(gamma) / safe_cosine**2
            if abs(cosine) >= 1e-3
            else 0.0
        )

        raw_next_gamma = gamma + vertical_acceleration / speed * dt
        gamma_limit = np.deg2rad(80.0)
        gamma_active = float(-gamma_limit < raw_next_gamma < gamma_limit)

        next_gamma = float(np.clip(raw_next_gamma, -gamma_limit, gamma_limit))
        next_heading = self.wrap_angle(
            heading + side_acceleration / (speed * safe_cosine) * dt
        )

        gamma_jacobian = np.array(
            [
                -np.sin(next_gamma) * np.cos(next_heading),
                -np.sin(next_gamma) * np.sin(next_heading),
                np.cos(next_gamma),
            ]
        )
        heading_jacobian = np.array(
            [
                -np.cos(next_gamma) * np.sin(next_heading),
                np.cos(next_gamma) * np.cos(next_heading),
                0.0,
            ]
        )

        next_gamma_gamma = gamma_active
        next_gamma_vertical = gamma_active * dt / speed
        next_heading_gamma = (
            side_acceleration / speed * dt * safe_cosine_derivative
        )
        next_heading_side = dt / (speed * safe_cosine)

        A = np.zeros((5, 5))
        B = np.zeros((5, 2))

        A[:3, :3] = np.eye(3)
        A[:3, 3] = speed * dt * (
            gamma_jacobian * next_gamma_gamma
            + heading_jacobian * next_heading_gamma
        )
        A[:3, 4] = speed * dt * heading_jacobian
        A[3, 3] = next_gamma_gamma
        A[4, 3] = next_heading_gamma
        A[4, 4] = 1.0

        B[:3, 0] = speed * dt * gamma_jacobian * next_gamma_vertical
        B[:3, 1] = speed * dt * heading_jacobian * next_heading_side
        B[3, 0] = next_gamma_vertical
        B[4, 1] = next_heading_side

        next_state = self.model(state, control, speed)
        affine = next_state - A @ state - B @ control

        return A, B, affine

    def rollout(self, initial_state, controls, speed):
        controls = np.asarray(
            controls,
            dtype=float,
        ).reshape(-1, 2)

        states = [
            np.asarray(
                initial_state,
                dtype=float,
            )
        ]

        for control in controls:
            states.append(
                self.model(
                    states[-1],
                    control,
                    speed,
                )
            )

        return np.asarray(states)

    def prediction_matrices(
        self,
        nominal_states,
        nominal_controls,
        speed,
    ):
        controls = np.asarray(
            nominal_controls,
            dtype=float,
        ).reshape(-1, 2)

        horizon = len(controls)
        control_columns = 2 * horizon

        S_k = np.zeros((5, control_columns))
        T_k = np.eye(5)
        h_k = np.zeros(5)

        S_blocks = []
        T_blocks = []
        h_blocks = []

        for step in range(horizon):
            A, B, affine = self.linearize(
                nominal_states[step],
                controls[step],
                speed,
            )

            S_k = A @ S_k
            column_start = 2 * step
            S_k[:, column_start : column_start + 2] += B
            T_k = A @ T_k
            h_k = A @ h_k + affine

            S_blocks.append(S_k.copy())
            T_blocks.append(T_k.copy())
            h_blocks.append(h_k.copy())

        S = np.vstack(S_blocks)
        T = np.vstack(T_blocks)
        h = np.concatenate(h_blocks)

        return S, T, h

    def cost_matrices(self, horizon, hit_index):
        cache_key = (int(horizon), int(hit_index))

        if cache_key in self.cost_cache:
            return self.cost_cache[cache_key]

        position_selector = np.diag(
            [
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
            ]
        )

        blocks = []

        for index in range(horizon):
            if index < hit_index:
                weight = self.tracking_weight
            elif index == hit_index:
                weight = self.hit_weight
            else:
                weight = 0.0

            blocks.append(weight * position_selector)

        Q = sparse.block_diag(
            blocks,
            format="csc",
        )

        normalized_control_weight = (
            self.control_weight / self.u_max**2
        )

        R = normalized_control_weight * np.eye(
            2 * horizon
        )

        self.cost_cache[cache_key] = (Q, R)

        return Q, R

    def control_bounds(self, horizon):
        horizon = int(horizon)

        if horizon in self.bound_cache:
            return self.bound_cache[horizon]

        axis_limit = self.u_max / np.sqrt(2.0)

        lower_bound = np.full(
            2 * horizon,
            -axis_limit,
        )
        upper_bound = np.full(
            2 * horizon,
            axis_limit,
        )

        self.bound_cache[horizon] = (
            lower_bound,
            upper_bound,
        )

        return lower_bound, upper_bound

    def control_expansion(self, horizon):
        horizon = int(horizon)

        if horizon in self.expansion_cache:
            return self.expansion_cache[horizon]

        block_count = int(
            np.ceil(horizon / self.control_block_size)
        )
        expansion = np.zeros(
            (
                2 * horizon,
                2 * block_count,
            )
        )

        for step in range(horizon):
            block = step // self.control_block_size
            row = slice(2 * step, 2 * step + 2)
            column = slice(2 * block, 2 * block + 2)
            expansion[row, column] = np.eye(2)

        self.expansion_cache[horizon] = expansion

        return expansion

    def expand_block_controls(
        self,
        block_controls,
        horizon,
    ):
        expansion = self.control_expansion(horizon)

        return (
            expansion @ np.asarray(block_controls).reshape(-1)
        ).reshape(horizon, 2)

    def solve_qp_problem(
        self,
        hessian,
        gradient,
        lower_bound,
        upper_bound,
    ):
        hessian = 0.5 * (
            hessian + hessian.T
        ) + 1e-8 * np.eye(len(gradient))

        self.qp_solve_count += 1

        try:
            solution = solve_qp(
                2.0 * hessian,
                2.0 * gradient,
                lb=lower_bound,
                ub=upper_bound,
                solver=self.qp_solver,
            )
        except ProblemError:
            solution = None

        if solution is None:
            self.solver_failures += 1

        return solution

    @staticmethod
    def desired_angles(relative_position):
        horizontal_distance = np.hypot(
            relative_position[0],
            relative_position[1],
        )

        flight_path_angle = np.arctan2(
            relative_position[2],
            horizontal_distance,
        )

        heading = np.arctan2(
            relative_position[1],
            relative_position[0],
        )

        return np.array(
            [
                flight_path_angle,
                heading,
            ]
        )

    @staticmethod
    def angle_error(desired, current):
        return np.arctan2(
            np.sin(desired - current),
            np.cos(desired - current),
        )

    def initial_control_seed(
        self,
        initial_state,
        target_position,
        horizon,
        speed,
    ):
        desired = self.desired_angles(
            target_position - initial_state[0:3]
        )

        error = self.angle_error(
            desired,
            initial_state[3:5],
        )

        total_time = max(
            horizon * self.dt,
            self.dt,
        )

        vertical_acceleration = (
            speed * error[0] / total_time
        )

        side_acceleration = (
            speed
            * self.safe_cos(initial_state[3])
            * error[1]
            / total_time
        )

        control = self.limit_control(
            np.array(
                [
                    vertical_acceleration,
                    side_acceleration,
                ]
            )
        )

        return np.tile(control, (horizon, 1))

    def endpoint_solution(
        self,
        initial_state,
        speed,
        target_position,
        initial_controls,
    ):
        controls = np.asarray(
            initial_controls,
            dtype=float,
        ).reshape(-1, 2).copy()

        horizon = len(controls)
        expansion = self.control_expansion(horizon)
        block_count = expansion.shape[1] // 2
        lower_bound, upper_bound = self.control_bounds(block_count)

        best_controls = controls.copy()
        best_miss = np.linalg.norm(
            self.rollout(
                initial_state,
                controls,
                speed,
            )[-1, 0:3]
            - target_position
        )

        for _ in range(self.reachability_iterations):
            nominal_states = self.rollout(
                initial_state,
                controls,
                speed,
            )

            S, T, h = self.prediction_matrices(
                nominal_states,
                controls,
                speed,
            )

            terminal_S = S[-5:, :]
            terminal_free = (
                T[-5:, :] @ initial_state + h[-5:]
            )

            position_S = terminal_S[0:3, :] @ expansion
            position_error = (
                terminal_free[0:3] - target_position
            )

            hessian = position_S.T @ position_S
            gradient = position_S.T @ position_error

            solution = self.solve_qp_problem(
                hessian,
                gradient,
                lower_bound,
                upper_bound,
            )

            if solution is None:
                break

            controls = self.expand_block_controls(
                solution,
                horizon,
            )

            nonlinear_terminal_position = self.rollout(
                initial_state,
                controls,
                speed,
            )[-1, 0:3]

            miss_distance = np.linalg.norm(
                nonlinear_terminal_position
                - target_position
            )

            if miss_distance < best_miss:
                best_miss = miss_distance
                best_controls = controls.copy()

        return float(best_miss), best_controls

    def prepare_targets(self, target_positions):
        target_positions = np.asarray(
            target_positions,
            dtype=float,
        )

        if (
            target_positions.ndim != 2
            or target_positions.shape[1] != 3
            or len(target_positions) == 0
        ):
            raise ValueError(
                "target_positions must have shape (steps, 3)."
            )

        return target_positions[: self.max_horizon]

    def estimate_interception(
        self,
        pursuer_state,
        target_positions,
    ):
        pursuer_state = np.asarray(
            pursuer_state,
            dtype=float,
        )
        target_positions = self.prepare_targets(
            target_positions
        )

        initial_state = pursuer_state[
            [
                0,
                1,
                2,
                4,
                5,
            ]
        ]
        speed = float(pursuer_state[3])

        feasible_indices = []

        for index, target_position in enumerate(target_positions):
            horizon = index + 1
            maximum_travel_distance = (
                speed * horizon * self.dt
                + self.intercept_radius
            )
            direct_distance = np.linalg.norm(
                target_position - initial_state[0:3]
            )

            if direct_distance <= maximum_travel_distance:
                feasible_indices.append(index)

        if not feasible_indices:
            feasible_indices = [len(target_positions) - 1]

        coarse_indices = feasible_indices[
            :: self.candidate_stride
        ]

        if coarse_indices[-1] != feasible_indices[-1]:
            coarse_indices.append(feasible_indices[-1])

        candidates = []
        previous_controls = None
        previous_coarse_index = feasible_indices[0] - 1

        for coarse_index in coarse_indices:
            minimum_miss, previous_controls = (
                self.evaluate_candidate(
                    initial_state,
                    speed,
                    target_positions[coarse_index],
                    coarse_index + 1,
                    previous_controls,
                )
            )

            candidates.append(
                (
                    coarse_index,
                    minimum_miss,
                    previous_controls.copy(),
                )
            )

            if minimum_miss <= self.intercept_radius:
                refinement_indices = [
                    index
                    for index in feasible_indices
                    if previous_coarse_index < index <= coarse_index
                ]

                refinement_controls = None

                for index in refinement_indices:
                    refined_miss, refinement_controls = (
                        self.evaluate_candidate(
                            initial_state,
                            speed,
                            target_positions[index],
                            index + 1,
                            refinement_controls,
                        )
                    )

                    if refined_miss <= self.intercept_radius:
                        return InterceptionEstimate(
                            j_hit=index + 1,
                            t_hit=(index + 1) * self.dt,
                            target_position=(
                                target_positions[index].copy()
                            ),
                            minimum_miss=refined_miss,
                            reachable=True,
                            fallback=None,
                        )

            previous_coarse_index = coarse_index

        best_index, best_miss, _ = min(
            candidates,
            key=lambda result: result[1],
        )

        return InterceptionEstimate(
            j_hit=best_index + 1,
            t_hit=(best_index + 1) * self.dt,
            target_position=target_positions[
                best_index
            ].copy(),
            minimum_miss=best_miss,
            reachable=False,
            fallback="minimum_terminal_miss",
        )

    def evaluate_candidate(
        self,
        initial_state,
        speed,
        target_position,
        horizon,
        previous_controls,
    ):
        if previous_controls is None:
            warm_start = np.zeros((horizon, 2))
        elif len(previous_controls) >= horizon:
            warm_start = previous_controls[:horizon].copy()
        else:
            warm_start = np.vstack(
                (
                    previous_controls,
                    np.repeat(
                        previous_controls[-1:],
                        horizon - len(previous_controls),
                        axis=0,
                    ),
                )
            )

        direct_start = self.initial_control_seed(
            initial_state,
            target_position,
            horizon,
            speed,
        )

        solutions = [
            self.endpoint_solution(
                initial_state,
                speed,
                target_position,
                seed,
            )
            for seed in (
                warm_start,
                direct_start,
            )
        ]

        return min(
            solutions,
            key=lambda result: result[0],
        )

    def shifted_controls(self, horizon):
        if self.U is None or len(self.U) != horizon:
            return np.zeros((horizon, 2))

        return np.vstack(
            (
                self.U[1:],
                self.U[-1],
            )
        )

    def solve(self, pursuer_state, target_positions):
        pursuer_state = np.asarray(
            pursuer_state,
            dtype=float,
        )
        target_positions = self.prepare_targets(
            target_positions
        )

        initial_state = pursuer_state[
            [
                0,
                1,
                2,
                4,
                5,
            ]
        ]
        speed = float(pursuer_state[3])
        previous_estimate = (
            None
            if self.last_info is None
            else self.last_info.get("estimate")
        )

        refresh_estimate = (
            previous_estimate is None
            or previous_estimate.j_hit <= 1
            or self.solve_count % self.estimate_update_interval == 0
        )

        if refresh_estimate:
            estimate = self.estimate_interception(
                pursuer_state,
                target_positions,
            )
        else:
            next_hit_step = max(
                previous_estimate.j_hit - 1,
                1,
            )
            estimate = InterceptionEstimate(
                j_hit=next_hit_step,
                t_hit=next_hit_step * self.dt,
                target_position=target_positions[
                    next_hit_step - 1
                ].copy(),
                minimum_miss=previous_estimate.minimum_miss,
                reachable=previous_estimate.reachable,
                fallback=previous_estimate.fallback,
            )

        self.solve_count += 1
        target_positions = target_positions[: estimate.j_hit]
        horizon = len(target_positions)

        nominal_controls = self.shifted_controls(
            horizon
        )
        nominal_states = self.rollout(
            initial_state,
            nominal_controls,
            speed,
        )

        S, T, h = self.prediction_matrices(
            nominal_states,
            nominal_controls,
            speed,
        )

        reference = np.zeros((horizon, 5))
        reference[:, 0:3] = target_positions
        reference = reference.reshape(-1)

        Q, R = self.cost_matrices(
            horizon,
            estimate.target_index,
        )

        error = T @ initial_state + h - reference
        expansion = self.control_expansion(horizon)
        blocked_S = S @ expansion
        blocked_R = expansion.T @ R @ expansion

        hessian = blocked_S.T @ (Q @ blocked_S) + blocked_R
        gradient = blocked_S.T @ (Q @ error)
        block_count = expansion.shape[1] // 2
        lower_bound, upper_bound = self.control_bounds(
            block_count
        )

        solution = self.solve_qp_problem(
            hessian,
            gradient,
            lower_bound,
            upper_bound,
        )

        if solution is None:
            self.U = nominal_controls
            solver_status = "fallback_warm_start"
        else:
            self.U = self.expand_block_controls(
                solution,
                horizon,
            )
            solver_status = "solved"

        self.last_prediction = self.rollout(
            initial_state,
            self.U,
            speed,
        )[1:]

        command = self.limit_control(self.U[0])

        self.last_info = {
            "estimate": estimate,
            "solver_status": solver_status,
            "control": command.copy(),
            "control_saturated": bool(
                np.linalg.norm(command) >= 0.99 * self.u_max
            ),
            "predicted_pursuer_trajectory": (
                self.last_prediction.copy()
            ),
            "control_block_size": self.control_block_size,
            "decision_variable_count": 2 * block_count,
            "qp_solve_count": self.qp_solve_count,
        }

        return command

