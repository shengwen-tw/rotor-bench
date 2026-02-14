import argparse
import gymnasium as gym
import numpy as np
import os
import random
import torch

from argparse import Namespace
from models.dynamics import DynamicsBatch
from models.quadrotor import QuadrotorEnv
from models.se3_math import TensorSE3
from models.esc import ESCBatch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def set_global_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Turn off for faster speed
    torch.backends.cudnn.benchmark = False


class QuadrotorVecEnv(VecEnv):
    def __init__(self, args, n_envs: int, training: bool, device: str):
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.num_envs = n_envs
        self.training = training
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(args.seed))

        # Create first environment to initialize DynamicsBatch
        env_args = Namespace(
            vehicle_cfg=args.vehicle_cfg,
            motion_cfg=args.motion_cfg,
            dt=args.dt,
            iterations=args.iterations,
            traj=args.traj,
            plan_yaw_traj="no",
            random_start="yes" if training else "no",
            renderer="offline",
            animate="no",
            plot="no",
            ctrl="RL",
        )
        _env = QuadrotorEnv(env_args, render_mode=None, rl_training=training)
        _env = Monitor(_env)
        _env.reset(seed=args.seed)
        observation_space = _env.observation_space
        action_space = _env.action_space

        # Invoke superclass constructor of the VecEnv class
        super().__init__(
            num_envs=self.num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        # Initialize DynamicsBatch for parallel environments rollout
        dt = _env.env.uav_dynamics.get_time_step()
        mass = _env.env.uav_dynamics.get_mass()
        _J = _env.env.uav_dynamics.get_inertia_matrix()
        _J = self.to_tensor(_J, dtype=self.dtype)
        J = _J.clone().expand(self.num_envs, 3, 3).contiguous()
        self.dynamics = DynamicsBatch(
            device=self.device,
            dt=dt,
            mass=mass,
            J=J,
            batch=self.num_envs,
        )

        # Total iterations
        self.iterations = int(_env.env.iterations)

        # Time index
        self.idx = self.new_0_tensor(self.num_envs, dtype=torch.long)

        # Desired trajectory
        self.xd = self.to_tensor(_env.env.xd, dtype=self.dtype)
        self.vd = self.to_tensor(_env.env.vd, dtype=self.dtype)
        self.yaw_d = self.new_0_tensor(
            self.num_envs, _env.env.iterations, dtype=self.dtype)

        # Current desired value (i.e., reference signal value)
        self.curr_xd = self.new_0_tensor(self.num_envs, 3, dtype=self.dtype)
        self.curr_vd = self.new_0_tensor(self.num_envs, 3, dtype=self.dtype)
        self.curr_yaw_d = self.new_0_tensor(self.num_envs, dtype=self.dtype)

        # First environment is no longer needed
        _env.close()

        # Attitude error definition (for reward shaping only)
        self.kx = torch.tensor([10.0, 10.0, 12.0],
                               device=self.device, dtype=self.dtype).view(1, 3)
        self.kv = torch.tensor([7.0, 7.0, 12.0],
                               device=self.device, dtype=self.dtype).view(1, 3)

        # Reward scales and weights
        self.pos_scale = 1.0
        self.vel_scale = 1.0
        self.att_scale = 0.5
        self.ang_scale = 2.0
        self.w_p = 1.0
        self.w_v = 0.3
        self.w_R = 0.3
        self.w_W = 0.1
        self.w_u = 0.01

        # Moment limits (N*m)
        self.M_max = torch.tensor([0.5, 0.5, 0.5],
                                  device=self.device, dtype=self.dtype).view(1, 3)

        # ESC model (RotorS-style parameters)
        self.esc_tau_up = 0.0125
        self.esc_tau_down = 0.025
        self.esc = ESCBatch(device=self.device, dtype=self.dtype,
                            Ct=8.54858e-06, batch=self.num_envs)

        # Reset async actions
        self.actions = None

        # Reset states
        idx_all_envs = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long)
        self.reset_envs(idx_all_envs)

    #=========#
    # Helpers #
    #=========#
    def to_tensor(self, x, dtype=None):
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    def new_0_tensor(self, *shape, dtype=None):
        return torch.zeros(*shape, device=self.device, dtype=dtype)

    def rand_tensor(self, size):
        return torch.rand(size, generator=self.rng,
                          device=self.device, dtype=self.dtype)

    #======================#
    # Gymnasium VecEnv API #
    #======================#
    def reset(self):
        with torch.no_grad():
            idx_all_envs = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long)
            self.reset_envs(idx_all_envs)
            obs_tensor = self.get_observation()
        return obs_tensor.detach().cpu().numpy()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        with torch.no_grad():
            # Get control moment and force from action
            actions = self.to_tensor(self.actions, dtype=self.dtype)
            M, f = self.execute_rl_action(actions)

            # Update quadrotor dyanmics
            self.dynamics.set_moment(M)
            self.dynamics.set_force(f)
            self.dynamics.update()

            # Advance time
            self.idx = self.idx + 1
            self.update_desired_state()

            # Compute observation, reward, and done
            obs = self.get_observation()
            reward, terminated, truncated, done = self.compute_reward()

            # Convert observation, reward, and done to Numpy for Stable-Baselines3
            obs_np = obs.detach().cpu().numpy()
            reward_np = reward.detach().cpu().numpy().astype(np.float32)
            done_np = done.detach().cpu().numpy().astype(bool)

            # Prepare infos array
            infos = [{} for _ in range(self.num_envs)]

            # Check if any environment is done
            if done.any():
                obs_done_np = obs_np.copy()

                # Find indices of done/terminated/truncated environments
                done_idx = done.nonzero(as_tuple=False).squeeze(-1)
                term_idx = terminated.nonzero(as_tuple=False).squeeze(-1)
                trunc_idx = truncated.nonzero(as_tuple=False).squeeze(-1)

                # Convert indices mask to set for looping
                term_set = set(term_idx.detach().cpu().tolist())
                trunc_set = set(trunc_idx.detach().cpu().tolist())

                # Fill info fields
                for i in done_idx.detach().cpu().tolist():
                    infos[i]["terminal_observation"] = obs_done_np[i]
                    infos[i]["TimeLimit.truncated"] = \
                        (i in trunc_set) and (i not in term_set)

                # Reset environments
                self.reset_envs(done_idx)

                # Return post-reset observation
                obs_post_reset_np = self.get_observation().detach().cpu().numpy()
                done_idx_np = done_idx.detach().cpu().numpy()
                obs_np[done_idx_np] = obs_post_reset_np[done_idx_np]

            # Reset async actions
            self.actions = None

            # Return observation, reward, done, and infos for Stable-Baselines3
            return obs_np, reward_np, done_np, infos

    def close(self):
        return

    def get_attr(self, attr_name, indices=None):
        indices = range(self.num_envs) if indices is None else indices
        return [getattr(self, attr_name) for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        indices = range(self.num_envs) if indices is None else indices
        for _ in indices:
            setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    #====================================#
    # Reinforcement learning environment #
    #====================================#
    @torch.no_grad()
    def reset_envs(self, idx: torch.Tensor):
        # Check if indices mask is empty or not
        if idx.numel() == 0:
            return

        # Reset time index
        self.idx[idx] = 0

        # Reset states (TODO)
        self.dynamics.x[idx] = 0.0
        self.dynamics.v[idx] = 0.0
        self.dynamics.W[idx] = 0.0
        self.dynamics.a[idx] = 0.0
        self.dynamics.W_dot[idx] = 0.0
        self.dynamics.R[idx] = torch.eye(
            3, device=self.device, dtype=self.dtype)
        self.esc.omega[idx] = 0.0
        self.esc.rotor_rpm[idx] = 0.0
        self.esc.cmd_rpm[idx] = 0.0
        self.esc.initialized[idx] = False

        # Randomize states
        if self.training:
            POS_INC_MAX = 1.5

            # Randomuze initial position
            pos_noise = 2.0 * self.rand_tensor((idx.numel(), 3)) - 1.0
            self.dynamics.x[idx] = self.dynamics.x[idx] + \
                pos_noise * float(POS_INC_MAX)

            # Randomize initial yaw angle
            yaw_rand = (2.0 * self.rand_tensor((idx.numel(),)) -
                        1.0) * torch.pi  # [-pi, pi]
            R_yaw_rand = TensorSE3.euler_to_rotmat(
                torch.zeros_like(yaw_rand),
                torch.zeros_like(yaw_rand),
                yaw_rand
            )
            self.dynamics.R[idx] = R_yaw_rand

            # Align yaw trajectory with the randomized yaw angle
            self.yaw_d[idx, :] = yaw_rand[:, None]

        # Refresh desired state
        self.update_desired_state()

    @torch.no_grad()
    def update_desired_state(self):
        idx = torch.clamp(self.idx, 0, self.iterations - 1)  # FIXME
        env_ids = torch.arange(self.num_envs, device=idx.device)
        self.curr_xd = self.xd[:, idx].transpose(0, 1).contiguous()
        self.curr_vd = self.vd[:, idx].transpose(0, 1).contiguous()
        #self.curr_ad = self.ad[:, idx].transpose(0, 1).contiguous()
        self.curr_yaw_d = self.yaw_d[env_ids, idx]
        #self.curr_Wd = self.Wd[idx].contiguous()
        #self.curr_W_dot_d = self.W_dot_d[idx].contiguous()

    @torch.no_grad()
    def get_observation(self) -> torch.Tensor:
        """Return observation for reinforcement learning."""
        x = self.dynamics.get_position()
        v = self.dynamics.get_velocity()
        R = self.dynamics.get_rotmat()
        Rt = R.transpose(1, 2)
        ex = x - self.curr_xd
        ev = v - self.curr_vd
        ex_b = (Rt @ ex.unsqueeze(-1)).squeeze(-1)
        ev_b = (Rt @ ev.unsqueeze(-1)).squeeze(-1)
        # Use body-frame x/y, but world-frame z for better altitude learning.
        ex_b[:, 2] = ex[:, 2]
        ev_b[:, 2] = ev[:, 2]
        euler = TensorSE3.rotmat_to_euler(R)
        W = self.dynamics.get_angular_velocity()
        eR = self._compute_attitude_error(ex, ev, R)
        return torch.cat([ex_b, ev_b, euler, W, eR], dim=1).to(self.dtype)

    @torch.no_grad()
    def compute_reward(self):
        # Compute reward
        x = self.dynamics.get_position()
        v = self.dynamics.get_velocity()
        R = self.dynamics.get_rotmat()
        Rt = R.transpose(1, 2)
        ex = x - self.curr_xd
        ev = v - self.curr_vd
        ex_b = (Rt @ ex.unsqueeze(-1)).squeeze(-1)
        ev_b = (Rt @ ev.unsqueeze(-1)).squeeze(-1)
        eR = self._compute_attitude_error(ex, ev, R)
        W = self.dynamics.get_angular_velocity()

        norm_ex = torch.linalg.norm(ex_b, dim=1) / self.pos_scale
        norm_ev = torch.linalg.norm(ev_b, dim=1) / self.vel_scale
        norm_eR = torch.linalg.norm(eR, dim=1) / self.att_scale
        norm_W = torch.linalg.norm(W, dim=1) / self.ang_scale

        u_term = 0.0
        if self.actions is not None:
            u = torch.as_tensor(
                self.actions, device=self.device, dtype=self.dtype)
            u_term = torch.linalg.norm(u[:, 0:3], dim=1)

        reward = -(
            self.w_p * norm_ex +
            self.w_v * norm_ev +
            self.w_R * norm_eR +
            self.w_W * norm_W +
            self.w_u * u_term
        )

        # Check termination
        terminated = (norm_ex > 10.0) | (norm_ev > 30.0)

        # Check truncation
        truncated = self.idx >= self.iterations

        done = terminated | truncated
        return reward, terminated, truncated, done

    @torch.no_grad()
    def execute_rl_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # States and parameters
        mass = self.dynamics.get_mass()
        g = self.dynamics.get_gravitational_acceleration()
        R = self.dynamics.get_rotmat()
        b3 = R[:, :, 2]

        # Reinfocement learning actions
        mx_cmd = action[:, 0]
        my_cmd = action[:, 1]
        mz_cmd = action[:, 2]
        thrust_cmd = action[:, 3]
        hover = mass * g
        # Map normalized thrust in [0, 1] to physical thrust [0, 3*hover]
        thrust_cmd = torch.clamp(thrust_cmd, 0.0, 1.0) * (3.0 * hover)

        # Map normalized moments in [-1, 1] to physical moments
        uav_ctrl_M = torch.clamp(
            torch.stack([mx_cmd, my_cmd, mz_cmd], dim=1), -1.0, 1.0
        ) * self.M_max

        # ESC lag: thrust command -> rotor speed -> thrust
        self.esc.update_from_thrust_command(
            thrust_cmd, dt=self.dynamics.get_time_step(),
            tau_up=self.esc_tau_up, tau_down=self.esc_tau_down)
        thrust_out = self.esc.get_esc_output_force()

        # Control force
        uav_ctrl_f = thrust_out.unsqueeze(1) * b3

        return uav_ctrl_M, uav_ctrl_f

    @torch.no_grad()
    def _compute_attitude_error(self, ex: torch.Tensor, ev: torch.Tensor,
                                R: torch.Tensor) -> torch.Tensor:
        # Desired thrust vector in world frame (for reward shaping)
        mass = self.dynamics.get_mass()
        g = self.dynamics.get_gravitational_acceleration()
        e3 = torch.tensor([0.0, 0.0, 1.0], device=self.device,
                          dtype=self.dtype).view(1, 3)
        f_n = -(-self.kx * ex - self.kv * ev - mass * g * e3 + mass * 0.0)
        norm_f = torch.linalg.norm(f_n, dim=1, keepdim=True)
        b3d = torch.where(norm_f > 1e-6, f_n / norm_f, e3)
        b1d = torch.stack([
            torch.cos(self.curr_yaw_d),
            torch.sin(self.curr_yaw_d),
            torch.zeros_like(self.curr_yaw_d),
        ], dim=1)
        b2d = torch.cross(b3d, b1d, dim=1)
        b1d_proj = torch.cross(b2d, b3d, dim=1)
        Rd = torch.stack([b1d_proj, b2d, b3d], dim=2)
        Rt = R.transpose(1, 2)
        Rdt = Rd.transpose(1, 2)
        eR = 0.5 * TensorSE3.vee_map_3x3(Rdt @ R - Rt @ Rd)
        return eR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vehicle_cfg', type=str,
                        default='quadrotor_f450.yaml')
    parser.add_argument('--motion_cfg', type=str,
                        default='motion_normal.yaml')
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--traj", type=str, default="HOVERING")
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=1000000)
    parser.add_argument("--logdir", type=str, default="runs/ppo_quadrotor")
    parser.add_argument("--checkpoint-every", type=int, default=200000)
    parser.add_argument("--tb", type=str, default="ppo_tb")
    parser.add_argument("--env-device", type=str, default="cuda")
    parser.add_argument("--ppo-device", type=str, default="cpu")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    set_global_seeds(args.seed)

    # Build training environment
    train_env = QuadrotorVecEnv(
        args, n_envs=args.n_envs, training=True, device=args.env_device)

    # Build evalution environment
    eval_args = argparse.Namespace(
        vehicle_cfg=args.vehicle_cfg,
        motion_cfg=args.motion_cfg,
        dt=args.dt,
        iterations=args.iterations,
        traj=args.traj,
        random_start=False,
        seed=args.seed + 10,
        n_envs=1,
        logdir=args.logdir
    )
    eval_env = QuadrotorVecEnv(
        eval_args, n_envs=1, training=False, device=args.env_device)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best"),
        log_path=os.path.join(args.logdir, "eval"),
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Train MLP with PPO
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=args.logdir,
        seed=args.seed,
        verbose=1,
        device=args.ppo_device,
    )

    # Start training
    model.learn(total_timesteps=args.total_steps,
                callback=eval_callback,
                tb_log_name=args.tb)

    # Save final model
    final_path = os.path.join(args.logdir, "final_model")
    model.save(final_path)
    print(f"[OK] Saved model to: {final_path}")


if __name__ == "__main__":
    main()
