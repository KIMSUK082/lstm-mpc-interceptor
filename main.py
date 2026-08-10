from vehicle import Vehicle
from sim import Sim
import numpy as np


def main():
    g = 9.81
    missile = Vehicle(0, 0, v=150.0)
    target = Vehicle(-1500, 500, v=150.0)

    sim = Sim(missile, target, T_max=45.0)
    traj_m, traj_p = sim.simulation()
    sim.plot_traj(traj_m, traj_p)


if __name__ == "__main__":
    main()

## lstm은 위치를 예측하게 구성할것
