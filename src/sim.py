import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque


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
        intercept_radius=10.0,
    ):
        self.missile = missile
        self.target = target
        self.dt = float(dt)
        self.T_max = float(T_max)
        self.rng = random.Random(seed)
        self.predictor = predictor
        self.mpc = mpc
        self.observation_time = float(observation_time)
        self.intercept_radius = float(intercept_radius)

        if self.dt <= 0.0:
            raise ValueError("dt must be greater than zero.")
        if self.observation_time < 40 * self.dt:
            raise ValueError("observation_time must provide at least 40 samples.")
        if self.intercept_radius <= 0.0:
            raise ValueError("intercept_radius must be greater than zero.")

        self.current_mode = None
        self.move_start_time = 0.0
        self.move_end_time = 0.0

        self.bank = 0.0
        self.weave_period = 4.0

        self.prediction_origins = []

    def select_new_move(self, time):
        self.move_start_time = time

        self.current_mode = self.rng.choices(
            population=["straight", "turn", "weave"],
            weights=[0.33, 0.34, 0.33],
            k=1,
        )[0]

        self.bank = 0.0

        if self.current_mode == "straight":
            self.bank = 0.0
            duration = self.rng.uniform(2.0, 5.0)

        elif self.current_mode == "turn":
            direction = self.rng.choice([-1.0, 1.0])
            bank_deg = self.rng.uniform(50.0, 80.0)
            self.bank = direction * np.radians(bank_deg)
            duration = self.rng.uniform(2.0, 5.0)

        elif self.current_mode == "weave":
            amplitude_deg = self.rng.uniform(75.0, 80.0)
            direction = self.rng.choice([-1.0, 1.0])
            self.bank = direction * np.radians(amplitude_deg)
            self.weave_period = self.rng.uniform(6.0, 8.0)
            n_cycles = self.rng.randint(1, 2)
            duration = n_cycles * self.weave_period

        self.move_end_time = time + duration

    def get_command(self, time):
        if self.current_mode == "straight":
            bank_cmd = 0.0

        elif self.current_mode == "turn":
            bank_cmd = self.bank

        elif self.current_mode == "weave":
            weave_time = time - self.move_start_time

            bank_cmd = self.bank * np.sin(2.0 * np.pi * weave_time / self.weave_period)
            ## bank_cmd=bank_amplitude*sin(2pi/T*n)

        else:
            bank_cmd = 0.0

        return bank_cmd

    def dataset_sim(self):
        dataset = []
        dataset.append(self.target.get_pos_vel())

        time = 0.0
        step_count = 0
        self.select_new_move(time)

        while time < self.T_max:

            if time >= self.move_end_time:
                self.select_new_move(time)

            bank_cmd = self.get_command(time)

            self.target.set_bank_cmd(bank_cmd)
            self.target.step(self.dt)
            dataset.append(self.target.get_pos_vel())

            step_count += 1
            time = step_count * self.dt

        return np.array(dataset)

    def simulation(self):
        traj_m = []
        traj_p = []
        predictions = []

        history = deque(maxlen=40)
        history.append(self.target.get_pos_vel())

        observation_steps = int(np.ceil(self.observation_time / self.dt))
        observation_start = -observation_steps * self.dt
        self.select_new_move(observation_start)

        for step in range(observation_steps):
            observation_time = observation_start + step * self.dt
            if observation_time >= self.move_end_time:
                self.select_new_move(observation_time)

            bank_cmd = self.get_command(observation_time)
            self.target.set_bank_cmd(bank_cmd)
            self.target.step(self.dt)
            history.append(self.target.get_pos_vel())

        traj_p.append(self.target.get_state())
        traj_m.append(self.missile.get_state())

        self.prediction_origins = []

        step_count = 0

        while step_count * self.dt < self.T_max:
            time = step_count * self.dt

            if time >= self.move_end_time:
                self.select_new_move(time)

            bank_cmd = self.get_command(time)

            pred_abs = self.predictor.predict(history)
            alat_cmd = self.mpc.solve(self.missile.get_state(), pred_abs)
            predictions.append(pred_abs)
            self.prediction_origins.append(self.target.get_state()[0:2].copy())

            self.missile.update_alat(alat_cmd)

            previous_relative_pos = (
                self.target.get_state()[0:2] - self.missile.get_state()[0:2]
            )

            self.target.set_bank_cmd(bank_cmd)
            self.target.step(self.dt)
            self.missile.step(self.dt)
            traj_p.append(self.target.get_state())
            traj_m.append(self.missile.get_state())

            history.append(self.target.get_pos_vel())

            current_relative_pos = (
                self.target.get_state()[0:2] - self.missile.get_state()[0:2]
            )
            relative_change = current_relative_pos - previous_relative_pos
            change_sq = np.dot(relative_change, relative_change)
            if change_sq > 0.0:
                fraction = np.clip(
                    -np.dot(previous_relative_pos, relative_change) / change_sq,
                    0.0,
                    1.0,
                )
            else:
                fraction = 0.0

            closest_relative_pos = previous_relative_pos + fraction * relative_change
            step_min_distance = np.linalg.norm(closest_relative_pos)

            if step_min_distance <= self.intercept_radius:
                break

            step_count += 1

        return traj_m, traj_p, predictions

    def plot_traj(
        self,
        traj_m,
        traj_p,
        predictions=None,
        xlim=None,
        ylim=None,
    ):
        M = np.asarray(traj_m)
        T = np.asarray(traj_p)

        ## aspect가 equal이라 데이터 가로:세로 비율대로 figure를 잡아야
        ## 위아래 여백만 생기고 그림이 납작해지는 걸 막을 수 있음
        pts = np.vstack([M[:, 0:2], T[:, 0:2]])
        x_range = np.ptp(pts[:, 0])
        y_range = np.ptp(pts[:, 1])
        ratio = np.clip(x_range / max(y_range, 1e-6), 1.0, 3.5)
        fig, ax = plt.subplots(figsize=(4.0 * ratio, 5.0))

        ax.plot(
            M[:, 0],
            M[:, 1],
            color="#FF7A45",
            lw=1.8,
            label="Missile",
        )

        ax.plot(
            T[:, 0],
            T[:, 1],
            color="#35D0BA",
            lw=2.5,
            alpha=0.5,
            label="Target",
        )

        if predictions:
            prediction_path = np.asarray(
                [np.asarray(prediction)[0] for prediction in predictions]
            )
            ax.plot(
                prediction_path[:, 0],
                prediction_path[:, 1],
                color="#1F2933",
                lw=1.2,
                ls="--",
                zorder=5,
                label="LSTM prediction",
            )

        # 시작 위치
        ax.plot(M[0, 0], M[0, 1], "o", color="#FF7A45", ms=8)
        ax.plot(T[0, 0], T[0, 1], "o", color="#35D0BA", ms=8)

        # 마지막 위치
        ax.plot(
            M[-1, 0],
            M[-1, 1],
            "x",
            color="#FF7A45",
            ms=10,
            mew=2,
        )
        ax.plot(
            T[-1, 0],
            T[-1, 1],
            "x",
            color="#35D0BA",
            ms=10,
            mew=2,
        )

        ## xlim/ylim을 주면 그 값으로 고정, 안 주면 궤적에 맞춰 자동 확대
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")

        # X축과 Y축에서 1m의 길이가 동일하게 보이도록 설정
        ax.set_aspect("equal", adjustable="box")

        ax.grid(alpha=0.3)
        ## 궤적이 납작해서 범례가 그림을 가리므로 축 밖으로 뺌
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
        fig.tight_layout()
        plt.show()
