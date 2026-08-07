from vehicle import Vehicle
from sim import Sim
import numpy as np


def main():
    g = 9.81
    missile = Vehicle(v=150.0)
    target = Vehicle(v=150.0)

    missile.init_state(0.0, 0.0, 0.0)
    target.init_state(1000.0, 1000.0, np.deg2rad(190))
    sim = Sim(missile, target, T_max=5)
    traj_m, traj_p = sim.simulation()
    sim.plot_traj(traj_m, traj_p)


if __name__ == "__main__":
    main()
