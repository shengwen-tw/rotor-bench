import numpy as np
import torch

from models.se3_math import TensorSE3
from torch import Tensor


class DynamicsBatch:
    """
    A PyTorch-based rigid body dynamics engine supporting batched computation
    and GPU acceleration.
    """

    def __init__(self, device: str, dt: float, mass: float, J: Tensor, batch: int = 1):
        # Tensor
        self.B = batch
        self.device = torch.device(device)
        self.dtype = torch.float32
        ref = torch.empty((), device=self.device, dtype=self.dtype)

        # Identity matrix
        _I = torch.eye(3, device=self.device, dtype=self.dtype)  # (3, 3)
        self.I = _I.view(1, 3, 3).expand(self.B, 3, 3)  # (B, 3, 3)

        # Z-vector of the inertial frame
        self.e3 = ref.new_zeros((self.B, 3))
        self.e3[:, 2] = 1.0

        self.dt = dt                             # Time step size [s]
        self.mass = mass                         # Mass [kg]
        self.g = 9.8                             # Gravity [m/s^2]
        self.J = J.clone().contiguous()          # Inertia matrix
        self.J_inv = torch.linalg.inv(self.J)    # Inverse inertia matrix
        self.J_d = self._nonstandard_inertia(self.J)
        self.x = ref.new_zeros((self.B, 3))      # Position [m]
        self.v = ref.new_zeros((self.B, 3))      # Velocity [m/s]
        self.a = ref.new_zeros((self.B, 3))      # Acceleration [m/s^2]
        self.R = self.I.clone().contiguous()     # Rotation matrix
        self.W = ref.new_zeros((self.B, 3))      # Angular vel. [rad/s]
        self.W_dot = ref.new_zeros((self.B, 3))  # Angular accel. [rad/s^2]
        self.f = ref.new_zeros((self.B, 3))      # Control force [N]
        self.M = ref.new_zeros((self.B, 3))      # Control moment [N*m]

    #=========#
    # Setters #
    #=========#
    def set_time_step(self, dt: float):
        self.dt = dt

    def set_mass(self, mass: float):
        self.mass = mass

    def set_inertia_matrix(self, J: Tensor):
        self.J = J.clone().contiguous()
        self.J_inv = torch.linalg.inv(self.J)
        self.J_d = self._nonstandard_inertia(self.J)

    def set_position(self, x: Tensor):
        self.x = x.clone().contiguous()

    def set_velocity(self, v: Tensor):
        self.v = v.clone().contiguous()

    def set_acceleration(self, a: Tensor):
        self.a = a.clone().contiguous()

    def set_rotmat(self, R: Tensor):
        self.R = R.clone().contiguous()

    def set_angular_velocity(self, W: Tensor):
        self.W = W.clone().contiguous()

    def set_angular_acceleration(self, W_dot: Tensor):
        self.W_dot = W_dot.clone().contiguous()

    def set_force(self, f: Tensor):
        self.f = f.clone().contiguous()

    def set_moment(self, M: Tensor):
        self.M = M.clone().contiguous()

    #=========#
    # Getters #
    #=========#
    def get_time_step(self) -> float:
        return self.dt

    def get_mass(self) -> float:
        return self.mass

    def get_gravitational_acceleration(self) -> float:
        return self.g

    def get_inertia_matrix(self) -> Tensor:
        return self.J

    def get_position(self) -> Tensor:
        return self.x

    def get_velocity(self) -> Tensor:
        return self.v

    def get_acceleration(self) -> Tensor:
        return self.a

    def get_rotmat(self) -> Tensor:
        return self.R

    def get_angular_velocity(self) -> Tensor:
        return self.W

    def get_angular_acceleration(self) -> Tensor:
        return self.W_dot

    def get_force(self) -> Tensor:
        return self.f

    def get_moment(self) -> Tensor:
        return self.M

    #=========#
    # Helpers #
    #=========#
    def _nonstandard_inertia(self, J: Tensor) -> Tensor:
        """Return the nonstandard inertia used by the discrete Lagrangian."""
        trace = torch.diagonal(J, dim1=-2, dim2=-1).sum(dim=-1)
        return 0.5 * trace[:, None, None] * self.I - J

    def _so3_exp(self, rotation_vector: Tensor) -> Tensor:
        """Batched SO(3) exponential map evaluated with Rodrigues' formula."""
        theta_squared = (rotation_vector * rotation_vector).sum(
            dim=1, keepdim=True)
        eps = torch.finfo(rotation_vector.dtype).eps
        theta = torch.sqrt(theta_squared.clamp_min(eps))

        # The Taylor branches keep the map well conditioned near the origin.
        sinc = torch.where(
            theta_squared < 1e-6,
            1.0 - theta_squared / 6.0 + theta_squared * theta_squared / 120.0,
            torch.sin(theta) / theta.clamp_min(eps),
        )
        cosc = torch.where(
            theta_squared < 1e-6,
            0.5 - theta_squared / 24.0
            + theta_squared * theta_squared / 720.0,
            (1.0 - torch.cos(theta)) / theta_squared.clamp_min(eps),
        )

        rotation_hat = TensorSE3.hat_map_3x3(rotation_vector)
        return (self.I + sinc[:, :, None] * rotation_hat
                + cosc[:, :, None] * (rotation_hat @ rotation_hat))

    def _solve_relative_rotation(self, angular_momentum: Tensor) -> Tensor:
        """Solve the implicit LGVI discrete Legendre transform for ``F``.

        The relative attitude ``F`` is the solution in SO(3) of

            F J_d - J_d F^T = h hat(angular_momentum).

        A Lie-group Newton iteration is used, so every iterate remains on
        SO(3).  A fixed iteration count avoids device synchronizations for
        batched GPU simulations.
        """
        omega = (self.J_inv @ angular_momentum[:, :, None])[:, :, 0]
        F = self._so3_exp(self.dt * omega)
        target = self.dt * angular_momentum

        for _ in range(3):
            FJ_d = F @ self.J_d
            residual = TensorSE3.vee_map_3x3(
                FJ_d - FJ_d.transpose(1, 2)) - target
            trace = torch.diagonal(FJ_d, dim1=1, dim2=2).sum(dim=1)
            jacobian = trace[:, None, None] * self.I - FJ_d
            delta = torch.linalg.solve(
                jacobian, -residual[:, :, None])[:, :, 0]
            F = self._so3_exp(delta) @ F

        return F

    def dv_dt(self, f: Tensor) -> Tensor:
        return (self.mass * self.g * self.e3 - f) / self.mass

    def dW_dt(self, W: Tensor) -> Tensor:
        # JW = (B, 3, 3) @ (B, 3, 1) -> (B, 3, 1)
        JW = (self.J @ W[:, :, None])[:, :, 0]
        # WJW = (B, 3, 1) X (B, 3, 1)
        W_cross_JW = torch.cross(W, JW, dim=1)
        # Wdot = (B, 3, 3) @ ((B, 3, 1) - (B, 3, 1))
        Wdot = (self.J_inv @ (self.M - W_cross_JW)[:, :, None])[:, :, 0]
        return Wdot

    #=================#
    # Dynamics update #
    #=================#
    def update(self):
        """
        Update rigid body dynamics for one time step.

        A forced Lie group variational integrator advances the complete SE(3)
        state.  Force (in the inertial frame) and moment (in the body frame)
        are held constant over the step.  The rotational increment is obtained
        from the implicit discrete Euler equation and therefore lies on SO(3)
        without a projection or attitude coordinates.
        """
        W = self.W
        self.a = self.dv_dt(self.f)
        self.W_dot = self.dW_dt(W)

        # Translational discrete Euler-Lagrange equations.  With a held force
        # these reduce to the exact constant-acceleration update.
        self.x = self.x + self.v * self.dt + 0.5 * self.a * (self.dt ** 2)
        self.v = self.v + self.a * self.dt

        # Forced rotational discrete Euler-Lagrange equations.  The same body
        # moment is used at both endpoints under the zero-order-hold assumption.
        momentum = (self.J @ W[:, :, None])[:, :, 0]
        momentum_half = momentum + 0.5 * self.dt * self.M
        F = self._solve_relative_rotation(momentum_half)
        momentum_next = (
            F.transpose(1, 2) @ momentum_half[:, :, None]
        )[:, :, 0] + 0.5 * self.dt * self.M

        self.R = self.R @ F
        self.W = (self.J_inv @ momentum_next[:, :, None])[:, :, 0]


class Dynamics:
    """
    A NumPy wrapper for single rigid body simulation using DynamicsBatch.
    """

    def __init__(self, dt, mass, J=np.eye(3), device="cpu"):
        # Convert inertia matrix from numpy to tensor type
        J_tensor = torch.as_tensor(
            J, dtype=torch.float32, device=device).view(1, 3, 3)

        # Create DynamicsBatch object
        self._dynamics = DynamicsBatch(
            device=device, dt=dt, mass=mass, J=J_tensor, batch=1)

    #=========#
    # Helpers #
    #=========#
    def _vec3x1_to_tensor(self, vec: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(vec, dtype=self._dynamics.dtype, device=self._dynamics.device).view(1, 3)

    def _mat3x3_to_tensor(self, mat: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(mat, dtype=self._dynamics.dtype, device=self._dynamics.device).view(1, 3, 3)

    def _tensor_to_ndarray(self, T: torch.Tensor) -> np.ndarray:
        return T[0].cpu().numpy().copy()

    #=========#
    # Setters #
    #=========#
    def set_time_step(self, dt: float):
        self._dynamics.set_time_step(dt)

    def set_mass(self, mass: float):
        self._dynamics.set_mass(mass)

    def set_inertia_matrix(self, J: np.ndarray):
        J_tensor = self._mat3x3_to_tensor(J)
        self._dynamics.set_inertia_matrix(J_tensor)

    def set_position(self, x: np.ndarray):
        x_tensor = self._vec3x1_to_tensor(x)
        self._dynamics.set_position(x_tensor)

    def set_velocity(self, v: np.ndarray):
        v_tensor = self._vec3x1_to_tensor(v)
        self._dynamics.set_velocity(v_tensor)

    def set_acceleration(self, a: np.ndarray):
        a_tensor = self._vec3x1_to_tensor(a)
        self._dynamics.set_acceleration(a_tensor)

    def set_rotmat(self, R: np.ndarray):
        R_tensor = self._mat3x3_to_tensor(R)
        self._dynamics.set_rotmat(R_tensor)

    def set_angular_velocity(self, W: np.ndarray):
        W_tensor = self._vec3x1_to_tensor(W)
        self._dynamics.set_angular_velocity(W_tensor)

    def set_angular_acceleration(self, W_dot: np.ndarray):
        W_dot_tensor = self._vec3x1_to_tensor(W_dot)
        self._dynamics.set_angular_acceleration(W_dot_tensor)

    def set_force(self, f: np.ndarray):
        f_tensor = self._vec3x1_to_tensor(f)
        self._dynamics.set_force(f_tensor)

    def set_moment(self, M: np.ndarray):
        M_tensor = self._vec3x1_to_tensor(M)
        self._dynamics.set_moment(M_tensor)

    #=========#
    # Getters #
    #=========#
    def get_time_step(self) -> float:
        return self._dynamics.get_time_step()

    def get_mass(self) -> float:
        return self._dynamics.get_mass()

    def get_gravitational_acceleration(self) -> float:
        return self._dynamics.get_gravitational_acceleration()

    def get_inertia_matrix(self) -> np.ndarray:
        J_tensor = self._dynamics.get_inertia_matrix()
        return self._tensor_to_ndarray(J_tensor)

    def get_position(self) -> np.ndarray:
        x_tensor = self._dynamics.get_position()
        return self._tensor_to_ndarray(x_tensor)

    def get_velocity(self) -> np.ndarray:
        v_tensor = self._dynamics.get_velocity()
        return self._tensor_to_ndarray(v_tensor)

    def get_acceleration(self) -> np.ndarray:
        a_tensor = self._dynamics.get_acceleration()
        return self._tensor_to_ndarray(a_tensor)

    def get_rotmat(self) -> np.ndarray:
        R_tensor = self._dynamics.get_rotmat()
        return self._tensor_to_ndarray(R_tensor)

    def get_angular_velocity(self) -> np.ndarray:
        W_tensor = self._dynamics.get_angular_velocity()
        return self._tensor_to_ndarray(W_tensor)

    def get_angular_acceleration(self) -> np.ndarray:
        W_dot_tensor = self._dynamics.get_angular_acceleration()
        return self._tensor_to_ndarray(W_dot_tensor)

    def get_force(self) -> np.ndarray:
        f_tensor = self._dynamics.get_force()
        return self._tensor_to_ndarray(f_tensor)

    def get_moment(self) -> np.ndarray:
        M_tensor = self._dynamics.get_moment()
        return self._tensor_to_ndarray(M_tensor)

    def state_randomize(self, np_random=None):
        POS_INC_MAX = 1.5
        VEL_INC_MAX = 1.5
        ANG_VEL_INC_MAX = 90  # [deg/s]

        rng = np_random if np_random is not None else np.random

        # position
        x = self.get_position()
        x += rng.uniform(-POS_INC_MAX, POS_INC_MAX, size=3)
        self.set_position(x)

        # v = self.get_velocity()
        # v += rng.uniform(-VEL_INC_MAX, VEL_INC_MAX, size=3)
        # self.set_velocity(v)

        # W = self.get_angular_velocity()
        # W += np.deg2rad(rng.uniform(-ANG_VEL_INC_MAX, ANG_VEL_INC_MAX, size=3))
        # self.set_angular_velocity(W)

    #=================#
    # Dynamics update #
    #=================#
    def update(self):
        self._dynamics.update()
