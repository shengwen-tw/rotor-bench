import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


class NMPCController:
    def __init__(self, args):
        self.iterations = args.iterations
        self.dt = args.dt

        self.Np = 20
        self.nx = 18
        self.nu = 4

        self.Qy = np.diag([400.0, 400.0, 750.0, 100.0, 100.0, 200.0, 0.5, 0.5, 0.2])
        self.Qu = np.diag([0.002, 0.02, 0.02, 0.01])
        self.KR = 0#1.0
        self.psi_weights = np.array([1.0, 1.0, 1.0])

        self.f_min = 0.0
        self.f_max = 25.0
        self.M_min = np.array([-10.0, -10.0, -10])
        self.M_max = np.array([10.0, 10.0, 10])

        self.solver = None
        self.last_solution = None
        self._build_solver()
        self.reset()

    def reset(self):
        self.idx = 0
        self.time_arr = np.zeros(self.iterations)
        self.cost_arr = np.zeros(self.iterations)
        self.psi_arr = np.zeros(self.iterations)
        self.ex_arr = np.zeros((3, self.iterations))
        self.ev_arr = np.zeros((3, self.iterations))
        self.W_arr = np.zeros((3, self.iterations))
        self.u_arr = np.zeros((4, self.iterations))
        self.last_solution = None

    @staticmethod
    def skew_np(w):
        return np.array([
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ])

    @staticmethod
    def skew_ca(w):
        return ca.vertcat(
            ca.horzcat(0.0, -w[2], w[1]),
            ca.horzcat(w[2], 0.0, -w[0]),
            ca.horzcat(-w[1], w[0], 0.0),
        )

    def cayley_np(self, w):
        I = np.eye(3)
        A = 0.5 * self.skew_np(w)
        return (I + A) @ np.linalg.inv(I - A)

    def cayley_ca(self, w):
        I = ca.DM.eye(3)
        A = 0.5 * self.skew_ca(w)
        return ca.mtimes(I + A, ca.inv(I - A))

    def pack_state(self, x, v, R, W):
        return np.concatenate([x, v, R.reshape(-1, order='F'), W])

    def unpack_state(self, state):
        x = state[0:3]
        v = state[3:6]
        R = state[6:15].reshape((3, 3), order='F')
        W = state[15:18]
        return x, v, R, W

    def rotation_error_np(self, R, Rd):
        psi_axes = np.zeros(3)
        for axis in range(3):
            e = np.zeros(3)
            e[axis] = 1.0
            psi_axes[axis] = 1.0 - np.dot(R @ e, Rd @ e)
        return float(np.dot(self.psi_weights, psi_axes))

    def rotation_error_ca(self, R, Rd):
        psi = 0
        for axis in range(3):
            psi += self.psi_weights[axis] * (1.0 - ca.dot(R[:, axis], Rd[:, axis]))
        return psi

    def observe_ca(self, state):
        return ca.vertcat(state[0:3], state[3:6], state[15:18])

    def discrete_update_ca(self, state, control, mass, J, g):
        x = state[0:3]
        v = state[3:6]
        R = ca.reshape(state[6:15], 3, 3)
        W = state[15:18]

        fc = control[0]
        M = control[1:4]
        e3 = ca.DM([0.0, 0.0, 1.0])

        x_next = x + v * self.dt
        v_dot = g * e3 - (fc / mass) * ca.mtimes(R, e3)
        v_next = v + v_dot * self.dt

        J_inv = ca.inv(J)
        W_dot = ca.mtimes(J_inv, -ca.cross(W, ca.mtimes(J, W)) + M)
        W_half = W + 0.5 * self.dt * W_dot
        R_next = ca.mtimes(R, self.cayley_ca(W_half * self.dt))
        W_next = W_half + 0.5 * self.dt * W_dot

        return ca.vertcat(x_next, v_next, ca.reshape(R_next, 9, 1), W_next)

    def build_reference_horizon(self, env):
        y_ref = []
        R_ref = []
        base_idx = env.idx

        for k in range(self.Np):
            ref_idx = min(base_idx + k, env.xd.shape[1] - 1)
            xd = env.xd[:, ref_idx]
            vd = env.vd[:, ref_idx]
            y_ref.append(np.concatenate([xd, vd, np.zeros(3)]))
            R_ref.append(np.eye(3).reshape(-1, order='F'))

        return np.concatenate(y_ref), np.concatenate(R_ref)

    def _build_solver(self):
        X = ca.SX.sym('X', self.nx, self.Np + 1)
        U = ca.SX.sym('U', self.nu, self.Np)

        x0_param = ca.SX.sym('x0', self.nx)
        y_ref_param = ca.SX.sym('y_ref', 9 * self.Np)
        R_ref_param = ca.SX.sym('R_ref', 9 * self.Np)
        mass_param = ca.SX.sym('mass')
        g_param = ca.SX.sym('g')
        J_param = ca.SX.sym('J', 3, 3)

        params = ca.vertcat(
            x0_param,
            y_ref_param,
            R_ref_param,
            mass_param,
            g_param,
            ca.reshape(J_param, 9, 1),
        )

        obj = 0
        g = [X[:, 0] - x0_param]

        for k in range(self.Np):
            yk = self.observe_ca(X[:, k])
            y_ref_k = y_ref_param[9 * k:9 * (k + 1)]
            Rk = ca.reshape(X[6:15, k], 3, 3)
            Rd_k = ca.reshape(R_ref_param[9 * k:9 * (k + 1)], 3, 3)
            uk = U[:, k]

            ey = yk - y_ref_k
            psi = self.rotation_error_ca(Rk, Rd_k)
            obj += ca.mtimes([ey.T, self.Qy, ey])
            obj += self.KR * psi * psi
            obj += ca.mtimes([uk.T, self.Qu, uk])

            x_next = self.discrete_update_ca(X[:, k], uk, mass_param, J_param, g_param)
            g.append(X[:, k + 1] - x_next)

        decision_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        constraints = ca.vertcat(*g)

        nlp = {'x': decision_vars, 'f': obj, 'g': constraints, 'p': params}
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 100,
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.lbg = np.zeros(constraints.shape[0])
        self.ubg = np.zeros(constraints.shape[0])

        x_block = np.full(self.nx * (self.Np + 1), -np.inf)
        x_block_upper = np.full(self.nx * (self.Np + 1), np.inf)
        u_lb = np.tile(np.concatenate([[self.f_min], self.M_min]), self.Np)
        u_ub = np.tile(np.concatenate([[self.f_max], self.M_max]), self.Np)
        self.lbx = np.concatenate([x_block, u_lb])
        self.ubx = np.concatenate([x_block_upper, u_ub])

    def build_initial_guess(self, x0, mass, g):
        if self.last_solution is None:
            X_guess = np.tile(x0.reshape(-1, 1), (1, self.Np + 1))
            hover_u = np.array([mass * g, 0.0, 0.0, 0.0])
            U_guess = np.tile(hover_u.reshape(-1, 1), (1, self.Np))
        else:
            X_prev = self.last_solution['X']
            U_prev = self.last_solution['U']
            X_guess = np.hstack([X_prev[:, 1:], X_prev[:, -1:]])
            U_guess = np.hstack([U_prev[:, 1:], U_prev[:, -1:]])
            X_guess[:, 0] = x0

        return np.concatenate([
            X_guess.reshape(-1, order='F'),
            U_guess.reshape(-1, order='F'),
        ])

    def solve_nmpc(self, x0, y_ref, R_ref, mass, J, g):
        params = np.concatenate([
            x0,
            y_ref,
            R_ref,
            np.array([mass, g]),
            J.reshape(-1, order='F'),
        ])

        init_guess = self.build_initial_guess(x0, mass, g)
        sol = self.solver(x0=init_guess, lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg, p=params)

        w_opt = np.array(sol['x']).flatten()
        x_size = self.nx * (self.Np + 1)
        X_opt = w_opt[:x_size].reshape((self.nx, self.Np + 1), order='F')
        U_opt = w_opt[x_size:].reshape((self.nu, self.Np), order='F')
        self.last_solution = {'X': X_opt, 'U': U_opt}
        return X_opt, U_opt, float(sol['f'])

    def run(self, env, record: bool = True):
        mass = env.uav_dynamics.get_mass()
        J = env.uav_dynamics.get_inertia_matrix()
        g = env.uav_dynamics.get_gravitational_acceleration()
        x = env.uav_dynamics.get_position()
        v = env.uav_dynamics.get_velocity()
        R = env.uav_dynamics.get_rotmat()
        W = env.uav_dynamics.get_angular_velocity()

        x0 = self.pack_state(x, v, R, W)
        y_ref, R_ref = self.build_reference_horizon(env)
        _, U_opt, total_cost = self.solve_nmpc(x0, y_ref, R_ref, mass, J, g)
        u = U_opt[:, 0]

        if record:
            ex = x - y_ref[0:3]
            ev = v - y_ref[3:6]
            psi = self.rotation_error_np(R, np.eye(3))
            self.time_arr[self.idx] = self.idx * self.dt
            self.cost_arr[self.idx] = total_cost
            self.psi_arr[self.idx] = psi
            self.ex_arr[:, self.idx] = ex
            self.ev_arr[:, self.idx] = ev
            self.W_arr[:, self.idx] = W
            self.u_arr[:, self.idx] = u
            self.idx += 1

        return np.array([u[0], u[1], u[2], u[3]])

    def plot_graph(self):
        plt.figure("NMPC control inputs")
        labels = ["f_c", "M_x", "M_y", "M_z"]
        for i in range(4):
            plt.subplot(4, 1, i + 1)
            plt.plot(self.time_arr, self.u_arr[i, :])
            plt.grid(True)
            plt.ylabel(labels[i])
        plt.xlabel("time [s]")

        plt.figure("NMPC objective")
        plt.plot(self.time_arr, self.cost_arr)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("cost")

        plt.figure("NMPC rotation error")
        plt.plot(self.time_arr, self.psi_arr)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("Psi(R)")

        plt.figure("NMPC angular velocity")
        labels = ["W_x", "W_y", "W_z"]
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(self.time_arr, np.rad2deg(self.W_arr[i, :]))
            plt.grid(True)
            plt.ylabel(labels[i] + " [deg/s]")
        plt.xlabel("time [s]")
