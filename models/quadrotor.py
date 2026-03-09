import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch

from models.dynamics import Dynamics
from models.esc import ESC, BatteryModel
from models.se3_math import NumpySE3
from models.thrust_allocator import ThrustAllocator
from control.geometric_control import GeometricMomentController
from viz.rigidbody_visualize import QuadRenderer
from gymnasium import spaces
from trajectory_planner import TrajectoryPlanner


class QuadrotorEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, args,
                 uav_dynamics=None, controller=None,
                 render_mode=None, rl_training: bool = False):
        super().__init__()
        self.args = args
        self.controller = controller
        self.iterations = args.iterations
        self.dt = args.dt
        self.render_mode = render_mode
        self.rl_training = rl_training

        # Load vehicle parameters from config file
        vehicle_path = 'configs/vehicles/' + args.vehicle_cfg
        with open(vehicle_path) as f:
            cfg = yaml.safe_load(f)
        multirotor_model = cfg["model"]
        mass = float(cfg["mass"])
        J = np.array(cfg["inertia"], dtype=np.float32)        
        arm_length = float(cfg["arm_length"])
        c_fau_f = float(cfg["thrust_to_torque"])
        esc_Ct = float(cfg["esc_Ct"])
        esc_tau_up = float(cfg["esc_tau_up"])
        esc_tau_down = float(cfg["esc_tau_down"])
        esc_v_max = cfg.get("esc_v_max", None)
        esc_v_min = cfg.get("esc_v_min", None)
        esc_v_k = float(cfg.get("esc_v_k", 0.0))
        esc_alpha = float(cfg.get("esc_alpha", 0.0))
        esc_battery_model = bool(cfg.get("esc_battery_model", True))
        self.esc_v_max = esc_v_max
        self.esc_v_min = esc_v_min
        self.esc_v_k = esc_v_k
        self.esc_alpha = esc_alpha
        self.esc_battery_model = esc_battery_model

        # Initialize quadrotor dynamics
        if uav_dynamics == None:
            self.uav_dynamics = Dynamics(dt=args.dt, mass=mass, J=J)
        else:
            self.uav_dynamics = uav_dynamics

        # Initialize trajectory planner
        traj_planner = TrajectoryPlanner(args)
        traj_planner.plan()

        # Initialize observation space for reinforcement learning
        # position error (3x1) + velocity error (3x1) + euler angles (3x1)
        pos_vel_low = np.full(6, -np.inf, dtype=np.float32)
        pos_vel_high = np.full(6, +np.inf, dtype=np.float32)
        euler_low = np.array([-np.pi, -np.pi/2, -np.pi], dtype=np.float32)
        euler_high = np.array([+np.pi, +np.pi/2, +np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([pos_vel_low, euler_low]),
            high=np.concatenate([pos_vel_high, euler_high]),
            dtype=np.float32
        )

        # Initialize action space for reinforcement learning
        ROLL_CTRL_MIN = -np.deg2rad(50)
        ROLL_CTRL_MAX = +np.deg2rad(50)
        PITCH_CTRL_MIN = -np.deg2rad(50)
        PITCH_CTRL_MAX = +np.deg2rad(50)
        mass = self.uav_dynamics.get_mass()
        g = self.uav_dynamics.get_gravitational_acceleration()
        hover = mass * g
        # Normalized thrust command in [0, 1] for RL
        THRUST_MIN = 0.0
        THRUST_MAX = 1.0
        self.action_space = spaces.Box(
            low=np.array([ROLL_CTRL_MIN, PITCH_CTRL_MIN,
                         THRUST_MIN], dtype=np.float32),
            high=np.array([ROLL_CTRL_MAX, PITCH_CTRL_MAX,
                          THRUST_MAX], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize moment controller for reinforcement learning
        self.moment_controller = GeometricMomentController()

        # Control allocation parameters
        self.thrust_allocator = ThrustAllocator(
            model=multirotor_model, d=arm_length, c_fau_f=c_fau_f
        )

        # Battery model
        if esc_battery_model:
            battery_model = BatteryModel(
                device=torch.device("cpu"),
                dtype=torch.float32,
                v_max=esc_v_max,
                v_min=esc_v_min,
                v_k=esc_v_k,
                alpha=esc_alpha,
            )
        else:
            battery_model = None

        # Electronic speed controller (ESC)
        if multirotor_model == "quadrotor":
            motor_count = 4
        else:
            raise ValueError(f"Unsupported model for ESCs: {multirotor_model}")
        self.escs = [ESC(Ct=esc_Ct, tau_up=esc_tau_up, tau_down=esc_tau_down,
                         battery=battery_model)
                     for _ in range(motor_count)]

        # Get random seed
        self.np_random, _ = gym.utils.seeding.np_random(None)

        # Reference signals
        self.xd = traj_planner.get_position_trajectory()  # Desired position
        self.vd = traj_planner.get_velocity_trajectory()  # Desired velocity
        self.ad = np.array([0.0, 0.0, 0.0])  # Desired acceleration
        self.yaw_d = traj_planner.get_yaw_trajectory()  # Desired yaw
        self.Wd = np.array([0.0, 0.0, 0.0])  # Desired angular velocity
        # Desired angular acceleration
        self.W_dot_d = np.array([0.0, 0.0, 0.0])

        # Initialize online renderer
        if args.renderer == 'online':
            self.viz = QuadRenderer.from_online(trajectory=self.xd)
        elif args.renderer == 'offline':
            self.viz = None
        else:
            raise ValueError(f"Unknown rendering type: {args.renderer}")

        # Reset states
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset time index
        self.idx = 0

        # Set random seed
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Set initial position and velocity
        self.uav_dynamics.set_position(self.xd[:, 0])
        self.uav_dynamics.set_velocity(self.vd[:, 0])

        # Set initial orientation (from Euler angles)
        roll = np.deg2rad(0)
        pitch = np.deg2rad(0)
        yaw = self.yaw_d[0]
        R = NumpySE3.euler_to_rotmat(roll, pitch, yaw)
        self.uav_dynamics.set_rotmat(R)

        # Randomize initial states
        if self.args.random_start == 'yes' or self.rl_training == True:
            self.uav_dynamics.state_randomize(self.np_random)

        # Reset controller
        if self.controller != None:
            self.controller.reset()

        # Data for plotting
        self.time_arr = np.zeros(self.iterations)
        self.accel_arr = np.zeros((3, self.iterations))
        self.vel_arr = np.zeros((3, self.iterations))
        self.pos_arr = np.zeros((3, self.iterations))
        self.R_arr = np.zeros((3, 3, self.iterations))
        self.euler_arr = np.zeros((3, self.iterations))
        self.W_dot_arr = np.zeros((3, self.iterations))
        self.W_arr = np.zeros((3, self.iterations))
        self.ideal_rpm_arr = np.zeros((len(self.escs), self.iterations))
        self.esc_rpm_arr = np.zeros((len(self.escs), self.iterations))
        self.voltage_arr = np.full(self.iterations, np.nan, dtype=float)
        self.eta_arr = np.full(self.iterations, np.nan, dtype=float)

        # Current desired values (i.e., reference signals)
        self.curr_xd = self.xd[:, 0]
        self.curr_vd = self.vd[:, 0]
        self.curr_ad = self.ad
        self.curr_yaw_d = self.yaw_d[0]
        self.curr_Wd = self.Wd
        self.curr_W_dot_d = self.W_dot_d
        self._reset_voltage_profile()

        # Return data for reinforcement learning
        obs = self.get_observation()
        return obs, {}

    def _reset_voltage_profile(self):
        if not self.esc_battery_model:
            return
        time_arr = np.arange(self.iterations, dtype=float) * self.dt
        voltage = self.esc_v_min + (self.esc_v_max - self.esc_v_min) * np.exp(
            -self.esc_v_k * time_arr
        )
        self.voltage_arr[:] = voltage
        if self.esc_alpha == 0.0:
            denom = (self.esc_v_max - self.esc_v_min)
            if denom == 0.0:
                self.eta_arr[:] = 0.0
            else:
                self.eta_arr[:] = (voltage - self.esc_v_min) / denom
        else:
            denom = np.exp(self.esc_alpha * (self.esc_v_max - self.esc_v_min)) - 1.0
            self.eta_arr[:] = (np.exp(self.esc_alpha * (voltage - self.esc_v_min)) - 1.0) / denom

    def reset_esc(self, action):
        if self.args.ctrl == 'RL':
            control_input = self.execute_rl_action(action)
        else:
            control_input = np.array(action, dtype=float)

        f_motors_cmd = self.thrust_allocator.motors_from_wrench(control_input)
        for i in range(len(self.escs)):
            self.escs[i].reset(float(f_motors_cmd[i]))

    def compute_reward(self):
        """Compute reward for reinforcement learning"""
        obs = self.get_observation()
        ex, ev = obs[0:3], obs[3:6]
        R = self.uav_dynamics.get_rotmat()
        Rt = R.T
        ex_b = Rt @ ex
        ev_b = Rt @ ev
        return -float(np.linalg.norm(ex_b) + 0.25*np.linalg.norm(ev_b))

    def get_observation(self):
        """Return observation for reinforcement learning"""
        x = self.uav_dynamics.get_position()
        v = self.uav_dynamics.get_velocity()
        R = self.uav_dynamics.get_rotmat()
        Rt = R.T
        ex = x - self.curr_xd
        ev = v - self.curr_vd
        ex_b = Rt @ ex
        ev_b = Rt @ ev
        euler = NumpySE3.rotmat_to_euler(R)
        return np.concatenate([ex_b, ev_b, euler]).astype(np.float32)

    def check_terminated(self):
        """Check terminaion for reinforcement learning"""
        if self.args.ctrl != 'RL':
            return False

        obs = self.get_observation()
        ex, ev = obs[0:3], obs[3:6]
        ex_too_large = np.linalg.norm(ex) > 10.0
        ev_too_large = np.linalg.norm(ev) > 30.0
        return ex_too_large or ev_too_large

    def check_truncated(self):
        return self.idx >= self.iterations

    def update_desired_state(self):
        self.curr_xd = self.xd[:, self.idx]
        self.curr_vd = self.vd[:, self.idx]
        self.curr_ad = self.ad
        self.curr_yaw_d = self.yaw_d[self.idx]
        self.curr_Wd = self.Wd
        self.curr_W_dot_d = self.W_dot_d

    def get_desired_state(self):
        return [self.curr_xd,
                self.curr_vd,
                self.curr_ad,
                self.curr_yaw_d,
                self.curr_Wd,
                self.curr_W_dot_d]

    def execute_rl_action(self, action):
        [roll_cmd, pitch_cmd, thrust_cmd] = action
        mass = self.uav_dynamics.get_mass()
        g = self.uav_dynamics.get_gravitational_acceleration()
        hover = mass * g
        # Map normalized thrust in [0, 1] to physical thrust [0, 3*hover]
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0) * (3.0 * hover)
        uav_ctrl_M = self.moment_controller.run(
            self.uav_dynamics, roll_cmd, pitch_cmd, self.curr_yaw_d)
        return np.array(
            [thrust_cmd, uav_ctrl_M[0], uav_ctrl_M[1], uav_ctrl_M[2]],
            dtype=float
        )

    def apply_esc(self, f_motors_cmd: np.ndarray):
        f_motors = np.zeros(len(self.escs), dtype=float)
        t = self.idx * self.dt
        for esc in self.escs:
            esc.set_time(t)
        for i in range(len(self.escs)):
            # Calculate RPM from thrust input and feed to the ESC model
            esc_rpm_cmd = self.escs[i].thrust_to_rpm(float(f_motors_cmd[i]))
            self.escs[i].set_target_force(float(f_motors_cmd[i]), self.dt)
            f_motors[i] = self.escs[i].get_actual_force()

            # Record data for plotting
            self.ideal_rpm_arr[i, self.idx] = esc_rpm_cmd
            self.esc_rpm_arr[i, self.idx] = self.escs[i].get_actual_rpm()
        return f_motors

    def step(self, action):
        """action: control input from controller or reinfocement learning"""

        if self.args.ctrl == 'RL':
            # For reinforcement learning, the action input is desired roll,
            # pitch, and thrust commands
            control_input = self.execute_rl_action(action)
        else:
            control_input = action

        # Calculate motor thrusts from the control input
        f_motors = self.thrust_allocator.motors_from_wrench(control_input)

        # ESC model # TODO: Add RL support
        if self.args.ctrl != 'RL':        
            f_motors = self.apply_esc(f_motors)

        # Real control input from ESC output to the multirotor dynamics
        control_input = self.thrust_allocator.wrench_from_motors(f_motors)

        # Convert control input into 3x1 force / moment vectors
        R = self.uav_dynamics.get_rotmat()
        uav_ctrl_M = control_input[1:4]
        uav_ctrl_f = control_input[0] * R[:, 2]

        # Update quadrotor dyanmics
        self.uav_dynamics.set_moment(uav_ctrl_M)
        self.uav_dynamics.set_force(uav_ctrl_f)
        self.uav_dynamics.update()

        # Record data for plotting
        self.time_arr[self.idx] = self.idx * self.dt
        self.accel_arr[:, self.idx] = self.uav_dynamics.get_acceleration()
        self.vel_arr[:, self.idx] = self.uav_dynamics.get_velocity()
        self.pos_arr[:, self.idx] = self.uav_dynamics.get_position()
        self.R_arr[:, :, self.idx] = self.uav_dynamics.get_rotmat()
        self.euler_arr[:, self.idx] = NumpySE3.rotmat_to_euler(
            self.uav_dynamics.get_rotmat())
        self.W_dot_arr[:,
                       self.idx] = self.uav_dynamics.get_angular_acceleration()
        self.W_arr[:, self.idx] = self.uav_dynamics.get_angular_velocity()

        # Update time index
        self.idx += 1

        # Update desired state from next
        truncated = self.check_truncated()
        if truncated == False:
            self.update_desired_state()

        # Return data for reinforcement learning
        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = self.check_terminated()
        return obs, reward, terminated, truncated, {}

    def plot_graph(self):
        if self.controller != None:
            self.controller.plot_graph()

        # Plot attitude (euler angles)
        plt.figure("Attitude (euler angles)")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, np.rad2deg(self.euler_arr[0, :]))
        plt.grid(True)
        plt.title("Attitude (euler angles)")
        plt.xlabel("time [s]")
        plt.ylabel("roll [deg]")
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, np.rad2deg(self.euler_arr[1, :]))
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("pitch [deg]")
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, np.rad2deg(self.euler_arr[2, :]), label="yaw")
        plt.plot(self.time_arr, np.rad2deg(self.yaw_d),
                 label="yaw_d", linestyle="--")
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("yaw [deg]")
        plt.legend()

        # Plot position (NED frame)
        plt.figure("Position (NED frame)")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, self.pos_arr[0, :], label="x[0]")
        plt.plot(self.time_arr, self.xd[0, :], label="xd[0]")
        plt.grid(True)
        plt.title("Position (NED frame)")
        plt.xlabel("time [s]")
        plt.ylabel("x [m]")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, self.pos_arr[1, :], label="x[1]")
        plt.plot(self.time_arr, self.xd[1, :], label="xd[1]")
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [m]")
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, -self.pos_arr[2, :], label="-x[2]")
        plt.plot(self.time_arr, -self.xd[2, :], label="-xd[2]")
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("-z [m]")
        plt.legend()

        # Plot velocity (NED frame)
        plt.figure("Velocity (NED frame)")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, self.vel_arr[0, :], label="v[0]")
        plt.plot(self.time_arr, self.vd[0, :], label="vd[0]")
        plt.grid(True)
        plt.title("Velocity (NED frame)")
        plt.xlabel("time [s]")
        plt.ylabel("x [m/s]")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, self.vel_arr[1, :], label="v[1]")
        plt.plot(self.time_arr, self.vd[1, :], label="vd[1]")
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [m/s]")
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, -self.vel_arr[2, :], label="-v[2]")
        plt.plot(self.time_arr, -self.vd[2, :], label="-vd[2]")
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("-z [m/s]")
        plt.legend()

        # Plot acceleration (NED frame)
        plt.figure("Acceleration (NED frame)")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, self.accel_arr[0, :])
        plt.grid(True)
        plt.title("Acceleration (NED frame)")
        plt.xlabel("time [s]")
        plt.ylabel("x [m/s^2]")
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, self.accel_arr[1, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [m/s^2]")
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, -self.accel_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("-z [m/s^2]")

        # 2D XY trajectory comparison
        plt.figure("XY Trajectory")
        plt.plot(self.xd[0, :], self.xd[1, :],
                 label="Desired Trajectory",
                 linestyle="--", linewidth=2)
        plt.plot(self.pos_arr[0, :], self.pos_arr[1, :],
                 label="True Position", alpha=0.8)
        plt.grid(True)
        plt.title("XY Trajectory")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis("equal")
        plt.legend()

        # Motor RPM (ideal vs ESC)
        plt.figure("Motor RPM")
        for i in range(4):
            plt.subplot(4, 1, i + 1)
            plt.plot(self.time_arr, self.ideal_rpm_arr[i, :], label=f"ideal_{i+1}")
            plt.plot(self.time_arr, self.esc_rpm_arr[i, :], label=f"esc_{i+1}")
            plt.grid(True)
            if i == 0:
                plt.title("Motor RPM (ideal vs ESC)")
            plt.xlabel("time [s]")
            plt.ylabel("rpm")
            plt.legend()

        # Voltage and efficiency
        if not np.isnan(self.voltage_arr).all():
            plt.figure("Voltage and Efficiency")
            plt.subplot(2, 1, 1)
            plt.plot(self.time_arr, self.voltage_arr)
            plt.grid(True)
            plt.title("Voltage and Efficiency")
            plt.xlabel("time [s]")
            plt.ylabel("voltage [V]")
            plt.subplot(2, 1, 2)
            plt.plot(self.time_arr, self.eta_arr)
            plt.grid(True)
            plt.xlabel("time [s]")
            plt.ylabel("eta [-]")

    def plot(self):
        if self.args.plot == 'yes':
            self.plot_graph()
        if self.args.animate == 'yes' and self.args.renderer == 'offline':
            self.render_offline()
        if self.args.plot == 'yes' or self.args.animate == 'yes':
            plt.show()

    def render_offline(self):
        self.viz = QuadRenderer.from_offline(self.pos_arr, self.R_arr,
                                             dt=self.dt,
                                             trajectory=self.xd)
        self.viz.animate()

    def render(self):
        skip = 10
        enabled = self.render_mode == 'human' or self.args.renderer == 'online'
        if self.idx % skip == 0 and enabled == True:
            R = self.uav_dynamics.get_rotmat()
            x = self.uav_dynamics.get_position()
            self.viz.render(R, x)
