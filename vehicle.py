import numpy as np


class Vehicle:
    def __init__(self, v, a_lat=0.0, a_long=0.0):
        self.v = v
        self.a_long = a_long  ##종가속도
        self.a_lat = a_lat  ##횡가속도

    def init_state(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta  ## 방향각

    def step(self, dt):
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.v += self.a_long * dt
        self.theta += self.a_lat / self.v * dt  ##방향 각속도

    def upadate_a_long(self, u):
        self.a_long = u

    def update_a_lat(self, u):
        self.a_lat = u

    def get_state(self):
        return np.array(
            [
                self.x,
                self.y,
                self.v,
                self.theta,
            ]
        )
