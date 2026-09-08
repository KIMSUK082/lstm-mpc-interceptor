import numpy as np

from missile import Missile
from model import TargetPredictor
from mpc import MPC
from sim import Sim
from vehicle import Vehicle


def main():
    target = Vehicle(
        -4000,
        -1500,
        v=150.0,
        heading=np.radians(90.0),
    )
    initial_heading = np.arctan2(target.y, target.x)
    missile = Missile(0, 0, v=400.0, head=initial_heading)
    predictor = TargetPredictor()
    mpc = MPC(dt=0.05, max_g=20.0)

    sim = Sim(
        target,
        missile,
        T_max=45.0,
        predictor=predictor,
        mpc=mpc,
        observation_time=2.0,
        maneuver_profile="dynamic",
        seed=36,
    )
    traj_m, traj_p, predictions = sim.simulation()
    sim.plot_traj(traj_m, traj_p, predictions)


if __name__ == "__main__":
    main()
