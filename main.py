from vehicle import Vehicle
from missile import Missile
from sim import Sim
import numpy as np
from model import TargetPredictor


def main():
    missile = Missile(0, 0, v=100)
    target = Vehicle(-1500, 500, v=150.0)
    predictor = TargetPredictor()

    sim = Sim(target, missile, T_max=45.0, predictor=predictor)
    traj_m, traj_p, prediction = sim.simulation()
    sim.plot_traj(traj_m, traj_p, prediction)


if __name__ == "__main__":
    main()
