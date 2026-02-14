import numpy as np
import torch


class ESCBatch:
    def __init__(self, device: torch.device, dtype: torch.dtype,
                 Ct: float = 9.339e-6, CR: float = 708.54,
                 omega_b: float = 155.83, batch: int = 1):
        # [N/(rad/s)^2]
        self.device = device
        self.dtype = dtype
        self.Ct = float(Ct)
        # Motor curve (steady-state): omega_ss = CR * sigma + omega_b
        self.CR = float(CR)          # [rad/s]
        self.omega_b = float(omega_b)  # [rad/s]
        self.omega = torch.zeros(batch, device=device, dtype=dtype)
        self.rotor_rpm = torch.zeros(batch, device=device, dtype=dtype)
        self.cmd_rpm = torch.zeros(batch, device=device, dtype=dtype)
        self.initialized = torch.zeros(batch, device=device, dtype=torch.bool)

    def update_from_force(self, thrust: torch.Tensor):
        # Approximate RPM from total thrust magnitude (single motor proxy).
        thrust = torch.clamp(thrust, min=0.0)
        self.omega = torch.sqrt(thrust / self.Ct)
        self.cmd_rpm = self.omega * (60.0 / (2.0 * torch.pi))
        self.rotor_rpm = self.cmd_rpm
        self.initialized[:] = True
        return self.rotor_rpm

    def update_from_throttle(self, sigma: torch.Tensor):
        sigma = torch.clamp(sigma, 0.0, 1.0)
        omega_ss = self.CR * sigma + self.omega_b
        self.cmd_rpm = omega_ss * (60.0 / (2.0 * torch.pi))
        self.omega = omega_ss
        self.rotor_rpm = self.cmd_rpm
        self.initialized[:] = True
        return self.rotor_rpm

    def update_from_thrust_command(self, thrust_cmd: torch.Tensor,
                                   dt: float, tau_up: float, tau_down: float):
        # First-order lag on rotor speed based on thrust command.
        thrust_cmd = torch.clamp(thrust_cmd, min=0.0)
        omega_ss = torch.sqrt(thrust_cmd / self.Ct)
        self.cmd_rpm = omega_ss * (60.0 / (2.0 * torch.pi))

        # On first command, match steady-state to simplify testing.
        not_init = ~self.initialized
        if not_init.any():
            self.omega = torch.where(not_init, omega_ss, self.omega)
            self.initialized = torch.ones_like(self.initialized)

        tau = torch.where(omega_ss >= self.omega,
                          torch.tensor(tau_up, device=self.device, dtype=self.dtype),
                          torch.tensor(tau_down, device=self.device, dtype=self.dtype))
        alpha = torch.clamp(torch.tensor(dt, device=self.device, dtype=self.dtype) / tau, max=1.0)
        self.omega = (1.0 - alpha) * self.omega + alpha * omega_ss
        self.rotor_rpm = self.omega * (60.0 / (2.0 * torch.pi))
        return self.rotor_rpm

    def thrust_from_rpm(self, rpm: torch.Tensor) -> torch.Tensor:
        omega = rpm * (2.0 * torch.pi / 60.0)
        return self.Ct * omega * omega

    def get_esc_output_force(self) -> torch.Tensor:
        return self.Ct * self.omega * self.omega


class ESC:
    def __init__(self, Ct: float = 9.339e-6, CR: float = 708.54,
                 omega_b: float = 155.83):
        self._batch = ESCBatch(device=torch.device("cpu"),
                               dtype=torch.float32,
                               Ct=Ct, CR=CR, omega_b=omega_b, batch=1)

    @property
    def rotor_rpm(self) -> float:
        return float(self._batch.rotor_rpm[0].item())

    @property
    def cmd_rpm(self) -> float:
        return float(self._batch.cmd_rpm[0].item())

    def update_from_force(self, thrust: float):
        return float(self._batch.update_from_force(torch.tensor([thrust]))[0].item())

    def update_from_throttle(self, sigma: float):
        return float(self._batch.update_from_throttle(torch.tensor([sigma]))[0].item())

    def update_from_thrust_command(self, thrust_cmd: float,
                                   dt: float, tau_up: float, tau_down: float):
        return float(self._batch.update_from_thrust_command(
            torch.tensor([thrust_cmd]), dt, tau_up, tau_down)[0].item())

    def thrust_from_rpm(self, rpm: float) -> float:
        return float(self._batch.thrust_from_rpm(torch.tensor([rpm]))[0].item())

    def get_esc_output_force(self) -> float:
        return float(self._batch.get_esc_output_force()[0].item())
