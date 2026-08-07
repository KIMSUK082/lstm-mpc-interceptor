import numpy as np
import matplotlib.pyplot as plt
import random


class Sim:
    def __init__(self, missile, target, dt=0.02, T_max=10.0):
        self.missile = missile
        self.target = target
        self.dt = dt
        self.T_max = T_max
        self.flag = 0

        self.min_switch_steps = round(0.8 / self.dt)
        self.max_switch_steps = round(1.2 / self.dt)

        self.next_switch_step = random.randint(
            self.min_switch_steps,
            self.max_switch_steps,
        )

    def mode(self, flag):
        g = 9.81

        if flag == 0:  # 직진
            return 0.0, 0.0

        elif flag == 1:  # 가속
            return 0.2 * g, 0.0

        elif flag == 2:  # 감속
            return -0.2 * g, 0.0

        elif flag == 3:  # 좌선회
            return 0.0, 4.0 * g

        elif flag == 4:  # 우선회
            return 0.0, -4.0 * g

    def simulation(self):
        step_num = int(self.T_max / self.dt)
        traj_m = []
        traj_p = []
        traj_p.append(self.target.get_state())
        traj_m.append(self.missile.get_state())

        for k in range(step_num):
            if k >= self.next_switch_step:
                self.flag = random.randint(0, 4)
                a_long, a_lat = self.mode(self.flag)
                self.target.upadate_a_long(a_long)
                self.target.update_a_lat(a_lat)

                switch_steps = random.randint(
                    self.min_switch_steps,
                    self.max_switch_steps,
                )

                self.next_switch_step += switch_steps

            self.target.step(self.dt)
            self.missile.step(self.dt)
            traj_p.append(self.target.get_state())
            traj_m.append(self.missile.get_state())
        return traj_m, traj_p

    def plot_traj(self, traj_m, traj_p):

        M = np.asarray(traj_m)
        T = np.asarray(traj_p)

        plt.figure(figsize=(7, 7))

        plt.plot(M[:, 0], M[:, 1], color="#FF7A45", lw=1.8, label="Missile")
        plt.plot(T[:, 0], T[:, 1], color="#35D0BA", lw=1.8, label="Target")

        plt.plot(M[0, 0], M[0, 1], "o", color="#FF7A45", ms=8)
        plt.plot(T[0, 0], T[0, 1], "o", color="#35D0BA", ms=8)
        plt.plot(M[-1, 0], M[-1, 1], "x", color="#FF7A45", ms=10, mew=2)
        plt.plot(T[-1, 0], T[-1, 1], "x", color="#35D0BA", ms=10, mew=2)

        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.gca().set_aspect("equal")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()
