import torch


class ESCBatch:
    def __init__(self, device: torch.device, dtype: torch.dtype,
                 Ct: float, tau_up: float, tau_down: float,
                 batch: int = 1):
        # PyTorch parameters
        self.device = device
        self.dtype = dtype

        # ESC model
        self.Ct = float(Ct)  # rotation-speed-to-thrust coefficient, [N/(rad/s)^2]
        self.tau_up = float(tau_up)     # Acceleration time constant, [s]
        self.tau_down = float(tau_down) # Deceleration time constant, [s]

        # Rotor speed state, [rad/s]
        self.omega = torch.zeros(batch, device=device, dtype=dtype)

    def reset(self, thrust: torch.Tensor):
        thrust = torch.clamp(thrust, min=0.0)
        self.omega = torch.sqrt(thrust / self.Ct)

    def set_target_force(self, thrust: torch.Tensor, dt: float):
        # Conver thurst command to RPM
        thrust = torch.clamp(thrust, min=0.0)
        omega_ss = torch.sqrt(thrust / self.Ct)

        # Check acceleration or deceleration
        tau = torch.where(omega_ss >= self.omega,
                          torch.tensor(self.tau_up, device=self.device, dtype=self.dtype),
                          torch.tensor(self.tau_down, device=self.device, dtype=self.dtype))

        # First-order low-pass filter
        dt_tensor = torch.tensor(dt, device=self.device, dtype=self.dtype)
        alpha = torch.clamp(dt_tensor / tau, max=1.0)
        self.omega = (1.0 - alpha) * self.omega + alpha * omega_ss

    def get_actual_force(self) -> torch.Tensor:
        return self.Ct * (self.omega ** 2)

    def get_actual_rpm(self) -> torch.Tensor:
        return self.omega * (60.0 / (2.0 * torch.pi))

    def thrust_to_rpm(self, thrust: torch.Tensor) -> torch.Tensor:
        thrust = torch.clamp(thrust, min=0.0)
        omega = torch.sqrt(thrust / self.Ct)
        return omega * (60.0 / (2.0 * torch.pi))


class ESC:
    def __init__(self, Ct: float, tau_up: float, tau_down: float):
        self._batch = ESCBatch(device=torch.device("cpu"),
                               dtype=torch.float32,
                               Ct=Ct, tau_up=tau_up, tau_down=tau_down, batch=1)

    def reset(self, thrust: float):
        self._batch.reset(torch.tensor([thrust]))

    def set_target_force(self, thrust: float, dt: float):
        self._batch.set_target_force(torch.tensor([thrust]), dt)

    def get_actual_force(self) -> float:
        return float(self._batch.get_actual_force()[0].item())

    def get_actual_rpm(self) -> float:
        return float(self._batch.get_actual_rpm()[0].item())

    def thrust_to_rpm(self, thrust: float) -> float:
        return float(self._batch.thrust_to_rpm(torch.tensor([thrust]))[0].item())
