import numpy as np
import math


class Vehicle:
    def __init__(self, x=0.0, y=0.0, v=100.0, heading=0.0, bank=0.0):
        self.x = x
        self.y = y
        self.v = v
        self.head = heading

        self.bank = bank
        self.bank_cmd = bank

        self.max_bank = math.radians(82.0)
        self.min_bank = -self.max_bank
        ## 현실에서는 90도가 가능한데 뭐 여러가지 제약이 있나봄 ㅈㄴ 어렵네

        self.roll_rate = 0.0
        self.max_roll_rate = math.radians(120)
        self.min_roll_rate = -self.max_roll_rate

        self.min_speed = 70.0
        self.max_speed = 220.0

        self.alat = 0.0
        self.head_rate = 0.0

        self.g = 9.81

        self.v_ref = 150.0
        self.k_i = 0.044
        self.a_thrust_max = 0.5 * self.g
        self.k_p = 0.05

    def update_bank(self, dt):
        bank_error = self.bank_cmd - self.bank
        max_bank_change = self.max_roll_rate * dt

        bank_change = np.clip(bank_error, -max_bank_change, max_bank_change)

        self.bank += bank_change

        self.bank = np.clip(
            self.bank,
            self.min_bank,
            self.max_bank,
        )

    ##(선택) 수정 필요 현재 구조는 bank 각도가 점진적으로 변하지 않고 이산적으로 변함

    def set_bank_cmd(self, bank_cmd):
        self.bank_cmd = np.clip(bank_cmd, self.min_bank, self.max_bank)

    def step(self, dt):

        self.update_bank(dt)
        self.alat = self.g * np.tan(self.bank)

        ## 비행기가 회전할때 일어나는 속도 변화
        n = 1.0 / np.cos(self.bank)
        a_drag = self.g * self.k_i * (n**2 - 1.0) * (self.v_ref / self.v) ** 2
        throttle = np.clip(self.k_p * (self.v_ref - self.v), 0.0, 1.0)
        a_thrust = throttle * self.a_thrust_max
        self.v = np.clip(
            self.v + (a_thrust - a_drag) * dt, self.min_speed, self.max_speed
        )
        ## llm 코드를 참고함
        self.head_rate = self.alat / np.clip(self.v, self.min_speed, self.max_speed)
        self.head += self.head_rate * dt
        self.x += self.v * np.cos(self.head) * dt
        self.y += self.v * np.sin(self.head) * dt

    def get_state(self):
        return np.array(
            [
                self.x,
                self.y,
                self.v,
                self.head,
                self.bank,
            ]
        )

    def get_pos_vel(self):
        return np.array(
            [self.x, self.y, self.v * np.cos(self.head), self.v * np.sin(self.head)]
        )
