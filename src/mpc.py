import numpy as np
from qpsolvers import solve_qp
from scipy import sparse


class MPC:
    def __init__(self, dt=0.05, max_g=40.0):
        self.dt = float(dt)
        self.max_g = float(max_g)
        self.u_max = self.max_g * 9.81

        self.Q = np.diag([1.0, 1.0, 0.0])
        self.Q_N = np.diag([20.0, 20.0, 0.0])
        self.R = np.array([[0.01]])

        self.U = None

    ## 미사일의 상태방정식
    def model(self, x_k, u_k, v):
        px, py, psi = x_k
        psi_next = psi + (u_k / v) * self.dt

        x_next = np.array(
            [
                px + v * np.cos(psi_next) * self.dt,
                py + v * np.sin(psi_next) * self.dt,
                psi_next,
            ]
        )

        return x_next

    ## 임의의 xk uk에서 선형화
    def linearize(self, x_k, u_k, v):
        psi_next = x_k[2] + (u_k / v) * self.dt

        A_k = np.array(
            [
                [1.0, 0.0, -v * self.dt * np.sin(psi_next)],
                [0.0, 1.0, v * self.dt * np.cos(psi_next)],
                [0.0, 0.0, 1.0],
            ]
        )

        B_k = np.array(
            [
                -self.dt**2 * np.sin(psi_next),
                self.dt**2 * np.cos(psi_next),
                self.dt / v,
            ]
        )

        x_next = self.model(x_k, u_k, v)
        d_k = x_next - A_k @ x_k - B_k * u_k

        return A_k, B_k, d_k

    ## mpc의 step 만큼의 xk 상태를 구함 선형화에 쓰일 임의의 점
    def rollout(self, x_0, U, v):
        X = [np.asarray(x_0, dtype=float)]

        for u_k in U:
            X.append(self.model(X[-1], u_k, v))

        return np.asarray(X)

    ## 선형화후 S T t를 구하는 함수
    def prediction_matrices(self, X, U, v):
        N = len(U)

        S_k = np.zeros((3, N))
        T_k = np.eye(3)
        t_k = np.zeros(3)

        S = []
        T = []
        t = []

        ## 매 스텝 마다 선형화
        for k in range(N):
            A_k, B_k, d_k = self.linearize(X[k], U[k], v)

            S_k = A_k @ S_k
            S_k[:, k] += B_k
            T_k = A_k @ T_k
            t_k = A_k @ t_k + d_k

            S.append(S_k.copy())
            T.append(T_k.copy())
            t.append(t_k.copy())

        S = np.vstack(S)
        T = np.vstack(T)
        t = np.concatenate(t)

        return S, T, t

    ## 가중치 행렬을 구함
    def cost_matrices(self, N):
        Q_blocks = [self.Q for _ in range(N - 1)] + [self.Q_N]
        Q = sparse.block_diag(
            Q_blocks, format="csc"
        )  ## 블록 대각 행렬 생성 csc는 희소행렬에서 0을 skip하여 메모리를 아끼는것
        R = sparse.kron(
            sparse.eye(N, format="csc"),
            sparse.csc_matrix(self.R),
            format="csc",  ## 크로네커 곱
        )

        return Q, R

    def solve(self, missile_state, target_positions):
        missile_state = np.asarray(missile_state, dtype=float)
        target_positions = np.asarray(target_positions, dtype=float)

        x_0 = missile_state[[0, 1, 3]]
        v = missile_state[2]
        N = len(target_positions)

        if self.U is None or len(self.U) != N:
            U = np.zeros(N)
        else:
            U = np.concatenate([self.U[1:], self.U[-1:]])

        X = self.rollout(x_0, U, v)
        S, T, t = self.prediction_matrices(X, U, v)

        X_ref = np.zeros((N, 3))
        X_ref[:, 0:2] = target_positions
        X_ref = X_ref.reshape(-1)

        Q, R = self.cost_matrices(N)

        e = T @ x_0 + t - X_ref
        P = S.T @ (Q @ S) + R.toarray()
        q = S.T @ (Q @ e)

        P = 0.5 * (P + P.T) + 1e-9 * np.eye(N)

        u_min = np.full(N, -self.u_max)
        u_max = np.full(N, self.u_max)

        self.U = solve_qp(
            P,
            q,
            lb=u_min,
            ub=u_max,
            solver="quadprog",
        )

        return float(self.U[0])
