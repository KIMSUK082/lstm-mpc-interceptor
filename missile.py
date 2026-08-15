import numpy as np


class Missile:
    def __init__(self, x=0.0, y=0.0, v=0.0, head=0.0):
        self.x = x
        self.y = y
        self.v = v
        self.head = head
        self.head_rate = 0.0
        self.alat = 0.0

    def step(self, dt):
        self.head_rate = self.alat / self.v
        self.head += self.head_rate * dt
        self.x += self.v * np.cos(self.head) * dt
        self.y += self.v * np.sin(self.head) * dt

    def update_alat(self, alat):
        self.alat = alat

    def get_state(self):
        return np.array([self.x, self.y, self.v, self.head, self.alat])
