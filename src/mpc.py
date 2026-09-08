import numpy as np
from qpsolvers import solve_qp


class MPC:

    def __init__(
        self,
        dt=0.05,
        max_g=20.0,
        disturbance_steps=8,
        los_velocity_scale=20.0,
        terminal_range_weight=20.0,
        control_weight=0.02,
        smooth_weight=0.05,
    ):
        self.dt = float(dt)
        self.max_g = float(max_g)
        self.u_max = self.max_g * 9.81
        self.disturbance_steps = int(disturbance_steps)

        ## 비용함수 가중치
        self.los_velocity_scale = float(los_velocity_scale)
        self.terminal_range_weight = float(terminal_range_weight)
        self.control_weight = float(control_weight)
        self.smooth_weight = float(smooth_weight)

        self.U = None
        self.last_u = 0.0

    """
        control: 추적 물체의 횡가속도
        speed: 추적 물체의 속력
        target_acceleration: LSTM이 예측한 가속도
    """

    def model(
        self,
        state,
        control,
        speed,
        target_acceleration,
    ):

        rx, ry, target_vx, target_vy, heading = state
        target_acceleration = np.array(target_acceleration)
        target_ax = target_acceleration[0]
        target_ay = target_acceleration[1]

        ## 표적의 다음 속도
        next_target_vx = target_vx + target_ax * self.dt
        next_target_vy = target_vy + target_ay * self.dt

        ## 추적 물체의 다음 진행 방향
        next_heading = heading + control / speed * self.dt

        ## 다음 상대 위치
        next_rx = rx + (next_target_vx - speed * np.cos(next_heading)) * self.dt
        next_ry = ry + (next_target_vy - speed * np.sin(next_heading)) * self.dt

        ## 다음 상태를 반환
        next_state = np.array(
            [
                next_rx,
                next_ry,
                next_target_vx,
                next_target_vy,
                next_heading,
            ]
        )

        return next_state

    def linearize(self, state, control, speed, target_acceleration):

        ## 기준점에서의 다음 진행 방향
        next_heading = state[4] + control / speed * self.dt

        ## 상태가 다음 상태에 미치는 영향
        A = np.array(
            [
                [
                    1.0,
                    0.0,
                    self.dt,
                    0.0,
                    speed * self.dt * np.sin(next_heading),
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    self.dt,
                    -speed * self.dt * np.cos(next_heading),
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            ]
        )

        ## 제어입력이 다음 상태에 미치는 영향
        B = np.array(
            [
                self.dt**2 * np.sin(next_heading),
                -self.dt**2 * np.cos(next_heading),
                0.0,
                0.0,
                self.dt / speed,
            ]
        )

        ## LSTM 가속도 예측이 다음 상태에 미치는 영향
        E = np.array(
            [
                [self.dt**2, 0.0],
                [0.0, self.dt**2],
                [self.dt, 0.0],
                [0.0, self.dt],
                [0.0, 0.0],
            ]
        )

        ## 정확한 비선형 모델의 계산 결과
        next_state = self.model(
            state,
            control,
            speed,
            target_acceleration,
        )

        ## 선형화 오차를 보정하는 상수항
        affine = next_state - A @ state - B * control - E @ target_acceleration

        return A, B, E, affine

    def rollout(
        self,
        initial_state,
        controls,
        speed,
        disturbances,
    ):

        initial_state = np.asarray(
            initial_state,
            dtype=float,
        )

        controls = np.asarray(
            controls,
            dtype=float,
        )

        disturbances = np.asarray(
            disturbances,
            dtype=float,
        )

        horizon = len(controls)

        ## 첫 번째 값은 현재 상태 x0
        nominal_states = [initial_state]

        for step in range(horizon):
            next_state = self.model(
                state=nominal_states[-1],
                control=controls[step],
                speed=speed,
                target_acceleration=disturbances[step],
            )

            nominal_states.append(next_state)

        return np.asarray(nominal_states)

    def prediction_matrices(
        self,
        nominal_states,
        controls,
        speed,
        disturbances,
    ):

        controls = np.asarray(
            controls,
            dtype=float,
        )

        disturbances = np.asarray(
            disturbances,
            dtype=float,
        )

        horizon = len(controls)

        ## 현재 스텝까지 누적된 행렬
        S_k = np.zeros((5, horizon))
        T_k = np.eye(5)
        h_k = np.zeros(5)

        ## 각 미래 스텝의 행렬을 저장할 리스트
        S_blocks = []
        T_blocks = []
        h_blocks = []

        for step in range(horizon):

            ## 현재 기준점에서 운동 모델을 선형화
            A, B, E, affine = self.linearize(
                state=nominal_states[step],
                control=controls[step],
                speed=speed,
                target_acceleration=disturbances[step],
            )

            S_k = A @ S_k
            S_k[:, step] += B
            T_k = A @ T_k
            h_k = A @ h_k + E @ disturbances[step] + affine

            S_blocks.append(S_k.copy())
            T_blocks.append(T_k.copy())
            h_blocks.append(h_k.copy())

        ## x1부터 xN까지 사용할 행렬을 세로로 쌓음
        S = np.vstack(S_blocks)
        T = np.vstack(T_blocks)
        h = np.concatenate(h_blocks)

        return S, T, h

    @staticmethod
    def los_velocity_linearization(
        state,
        speed,
    ):

        ## 선형화 기준 상태
        rx, ry, target_vx, target_vy, heading = state

        ## 상대거리
        distance = max(
            np.hypot(rx, ry),
            1e-6,
        )

        ## 추적 물체의 전역 X/Y 속도
        pursuer_vx = speed * np.cos(heading)
        pursuer_vy = speed * np.sin(heading)

        ## 표적 속도 - 추적 물체 속도
        relative_vx = target_vx - pursuer_vx
        relative_vy = target_vy - pursuer_vy

        ## V_lambda 식의 분자
        numerator = rx * relative_vy - ry * relative_vx

        ## 시선 수직 상대속도
        los_velocity = numerator / distance

        ## V_lambda를 각 상태변수로 미분한 결과
        gradient = np.array(
            [
                (relative_vy / distance - numerator * rx / distance**3),
                (-relative_vx / distance - numerator * ry / distance**3),
                -ry / distance,
                rx / distance,
                (-speed * (rx * np.cos(heading) + ry * np.sin(heading)) / distance),
            ]
        )

        offset = los_velocity - gradient @ state

        return gradient, offset

    def solve(
        self,
        pursuer_state,
        target_state,
        predicted_accelerations,
    ):

        pursuer_state = np.asarray(pursuer_state, dtype=float)
        target_state = np.asarray(target_state, dtype=float)

        ## LSTM 가속도 예측을 MPC 외란으로 사용
        disturbances = np.asarray(
            predicted_accelerations,
            dtype=float,
        )
        horizon = len(disturbances)

        ## 현재 상태 x0 생성
        speed = max(float(pursuer_state[2]), 1e-6)
        target_speed = target_state[2]
        target_heading = target_state[3]

        target_vx = target_speed * np.cos(target_heading)
        target_vy = target_speed * np.sin(target_heading)

        initial_state = np.array(
            [
                target_state[0] - pursuer_state[0],
                target_state[1] - pursuer_state[1],
                target_vx,
                target_vy,
                pursuer_state[3],
            ]
        )

        ## 이전 최적해를 한 스텝 앞으로 이동하여 기준 제어입력 생성
        if self.U is None or len(self.U) != horizon:
            controls = np.zeros(horizon)
        else:
            controls = np.concatenate([self.U[1:], self.U[-1:]])

        ## 선형화에 사용할 기준 궤적 생성
        nominal_states = self.rollout(
            initial_state,
            controls,
            speed,
            disturbances,
        )

        ## X = S U + T x0 + h
        S, T, h = self.prediction_matrices(
            nominal_states,
            controls,
            speed,
            disturbances,
        )

        free_states = T @ initial_state + h

        ## 미래 각 스텝의 V_lambda를 선형화
        los_rows = np.zeros((horizon, 5 * horizon))
        los_offsets = np.zeros(horizon)

        for step in range(horizon):
            gradient, offset = self.los_velocity_linearization(
                nominal_states[step + 1],
                speed,
            )

            los_rows[step, 5 * step : 5 * (step + 1)] = gradient
            los_offsets[step] = offset

        ## V_lambda = los_control U + los_free
        los_control = los_rows @ S
        los_free = los_rows @ free_states + los_offsets
        los_weight = 1.0 / self.los_velocity_scale**2

        ## 마지막 스텝의 상대위치 rx, ry만 선택
        terminal_selector = np.zeros((2, 5 * horizon))
        terminal_selector[:, -5:-3] = np.eye(2)

        range_control = terminal_selector @ S
        range_free = terminal_selector @ free_states

        initial_range = max(np.linalg.norm(initial_state[:2]), 100.0)
        range_weight = self.terminal_range_weight / initial_range**2

        ## 제어입력 변화량 계산을 위한 행렬
        difference = np.eye(horizon)
        difference[1:, :-1] -= np.eye(horizon - 1)

        difference_reference = np.zeros(horizon)
        difference_reference[0] = self.last_u

        control_scale_squared = self.u_max**2

        ## LOS 수직 상대속도 비용
        los_hessian = los_weight * los_control.T @ los_control
        los_gradient = los_weight * los_control.T @ los_free

        ## 마지막 상대거리 비용
        range_hessian = range_weight * range_control.T @ range_control
        range_gradient = range_weight * range_control.T @ range_free

        ## 제어입력 크기 비용
        control_hessian = self.control_weight / control_scale_squared * np.eye(horizon)

        ## 제어입력 변화량 비용
        smooth_hessian = (
            self.smooth_weight / control_scale_squared * difference.T @ difference
        )

        smooth_gradient = (
            -self.smooth_weight
            / control_scale_squared
            * difference.T
            @ difference_reference
        )

        ## 전체 비용함수를 QP 형태로 결합
        hessian = los_hessian + range_hessian + control_hessian + smooth_hessian

        gradient = los_gradient + range_gradient + smooth_gradient

        P = 2.0 * hessian
        q = 2.0 * gradient
        P = 0.5 * (P + P.T) + 1e-9 * np.eye(horizon)

        ## 횡가속도 제한 안에서 QP 계산
        self.U = solve_qp(
            P,
            q,
            lb=np.full(horizon, -self.u_max),
            ub=np.full(horizon, self.u_max),
            solver="quadprog",
        )

        ## 최적화 실패 시 기준 제어입력 사용
        if self.U is None:
            self.U = controls

        ## 첫 번째 제어입력만 실제 시스템에 적용
        self.last_u = float(np.clip(self.U[0], -self.u_max, self.u_max))

        return self.last_u
