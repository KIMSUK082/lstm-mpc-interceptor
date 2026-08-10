import numpy as np
import matplotlib.pyplot as plt
import random


class Sim:
    def __init__(self, missile, target, dt=0.02, T_max=15.0):
        self.missile = missile
        self.target = target
        self.dt = dt
        self.T_max = T_max

        self.current_mode = None
        self.move_start_time = 0.0
        self.move_end_time = 0.0

        self.bank = 0.0
        self.weave_period = 4.0

    def select_new_move(self, time):
        self.move_start_time = time

        self.current_mode = random.choices(
            population=["straight", "turn", "weave"],
            weights=[0.33, 0.34, 0.33],
            k=1,
        )[0]

        self.bank = 0.0

        if self.current_mode == "straight":
            self.bank = 0.0
            duration = random.uniform(2.0, 5.0)

        elif self.current_mode == "turn":
            direction = random.choice([-1.0, 1.0])
            bank_deg = random.uniform(50.0, 80.0)
            self.bank = direction * np.radians(bank_deg)
            duration = random.uniform(2.0, 5.0)

        elif self.current_mode == "weave":
            amplitude_deg = random.uniform(75.0, 80.0)
            direction = random.choice([-1.0, 1.0])
            self.bank = direction * np.radians(amplitude_deg)
            self.weave_period = random.uniform(6.0, 8.0)
            n_cycles = random.randint(1, 2)
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

    def simulation(self):
        traj_m = []
        traj_p = []
        traj_p.append(self.target.get_state())
        traj_m.append(self.missile.get_state())
        time = 0.0
        self.select_new_move(time)

        while time < self.T_max:

            if time >= self.move_end_time:
                self.select_new_move(time)

            bank_cmd = self.get_command(time)

            self.target.set_bank_cmd(bank_cmd)
            self.target.step(self.dt)
            traj_p.append(self.target.get_state())
            traj_m.append(self.missile.get_state())

            time += self.dt

        return traj_m, traj_p

    def plot_traj(
        self,
        traj_m,
        traj_p,
        xlim=(-5000, 5000),
        ylim=(-5000, 5000),
    ):
        M = np.asarray(traj_m)
        T = np.asarray(traj_p)

        fig, ax = plt.subplots(figsize=(7, 7))

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
            lw=1.8,
            label="Target",
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

        # X, Y축 범위 고정
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")

        # X축과 Y축에서 1m의 길이가 동일하게 보이도록 설정
        ax.set_aspect("equal", adjustable="box")

        ax.grid(alpha=0.3)
        ax.legend()
        plt.show()
