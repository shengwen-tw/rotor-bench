import numpy as np


class ThrustAllocator:
    def __init__(self, model: str, d: float, c_fau_f: float):
        if model != "quadrotor":
            raise ValueError(f"Unsupported model for ThrustAllocator: {model}")

        self.d = d
        self.c_fau_f = c_fau_f
        inv_4 = 0.25
        inv_4d = 1.0 / (4.0 * self.d)
        inv_4c = 1.0 / (4.0 * self.c_fau_f)

        self.T = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-self.d, self.d,  self.d,-self.d],
            [ self.d, self.d, -self.d, -self.d],
            [-self.c_fau_f, self.c_fau_f, -self.c_fau_f, self.c_fau_f],
        ])

        self.T_inv = np.array([
            [inv_4, -inv_4d,  inv_4d, -inv_4c],
            [inv_4,  inv_4d,  inv_4d,  inv_4c],
            [inv_4,  inv_4d, -inv_4d, -inv_4c],
            [inv_4, -inv_4d, -inv_4d,  inv_4c],
        ])

    def motors_from_wrench(self, wrench: np.ndarray):
        # wrench = [f_collective, Mx, My, Mz]
        return self.T_inv @ np.array(wrench)

    def wrench_from_motors(self, f_motors: np.ndarray):
        # f_motors = [motor1, motor2, motor3, motor4, ...]
        return self.T @ np.array(f_motors)
