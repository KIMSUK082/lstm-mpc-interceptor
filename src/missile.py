import numpy as np


class Missile:
    def __init__(self, x=0.0, y=0.0, v=500.0, head=0.0):
        self.x = float(x)
        self.y = float(y)
        self.v = float(v)
        self.head = float(head)
        self.alat = 0.0

    def step(self, dt):
        self.head += (self.alat / self.v) * dt
        self.x += self.v * np.cos(self.head) * dt
        self.y += self.v * np.sin(self.head) * dt

    def update_alat(self, alat):
        self.alat = float(alat)

    def get_state(self):
        return np.array([self.x, self.y, self.v, self.head])
