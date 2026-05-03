import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


class NMPC:
    def __init__(self, args):
        self.iterations = args.iterations
        self.dt = args.dt

        # Step for state horizon
        self.Np = 50

        # Dimension of the state
        self.x_dim = 18

        # Dimension of the control input
        self.u_dim = 4

        # Dimension of the tracking error xi = [ex, ev, eR, eW]
        self.xi_dim = 12

        # State error penalty weights
        self.Qy = np.zeros((self.xi_dim, self.xi_dim))
        self.Qy[0, 0] = 1500.0  # px
        self.Qy[1, 1] = 1500.0  # py
        self.Qy[2, 2] = 750.0   # pz
        self.Qy[3, 3] = 200.0   # vx
        self.Qy[4, 4] = 250.0   # vy
        self.Qy[5, 5] = 250.0   # vz
        self.Qy[6, 6] = 1.0     # eRx
        self.Qy[7, 7] = 1.0     # eRy
        self.Qy[8, 8] = 1.0     # eRz
        self.Qy[9, 9] = 0.5     # Wx
        self.Qy[10, 10] = 0.5   # Wy
        self.Qy[11, 11] = 0.5   # Wz

        # Control penalty weights
        self.Qu = np.zeros((4, 4))
        self.Qu[0, 0] = 0.002  # fc
        self.Qu[1, 1] = 0.02   # Mx
        self.Qu[2, 2] = 0.02   # My
        self.Qu[3, 3] = 0.01   # Mz

        # Constraints for control imputs
        self.f_min = 0.0
        self.f_max = 25.0
        self.M_min = np.array([-10.0, -10.0, -10])
        self.M_max = np.array([+10.0, +10.0, +10])

        # CasADi solver
        self.solver = None
        self.last_solution = None
        self.build_solver()

        # Reset data
        self.reset()


    def reset(self):
        self.idx = 0
        self.time_arr = np.zeros(self.iterations)
        self.cost_arr = np.zeros(self.iterations)
        self.ex_arr = np.zeros((3, self.iterations))
        self.ev_arr = np.zeros((3, self.iterations))
        self.eW_arr = np.zeros((3, self.iterations))
        self.W_arr = np.zeros((3, self.iterations))
        self.M_arr = np.zeros((3, self.iterations))
        self.f_arr = np.zeros(self.iterations)
        self.u_arr = np.zeros((4, self.iterations))
        self.last_solution = None

    def skew_sym_matrix(self, w):
        return ca.vertcat(
            ca.horzcat(0.0, -w[2], w[1]),
            ca.horzcat(w[2], 0.0, -w[0]),
            ca.horzcat(-w[1], w[0], 0.0),
        )

    def cayley_transform(self, w):
        I = ca.DM.eye(3)
        A = 0.5 * self.skew_sym_matrix(w)
        return ca.solve(I - A, I + A)

    def pack_state(self, x, v, R, W):
        return np.concatenate([x, v, R.reshape(-1, order='F'), W])

    def unpack_state(self, state):
        x = state[0:3]
        v = state[3:6]
        R = state[6:15].reshape((3, 3), order='F')
        W = state[15:18]
        return x, v, R, W

    def vee_map(self, M):
        return ca.vertcat(M[2, 1], M[0, 2], M[1, 0])

    def attitude_error(self, R, Rd):
        e_R_mat = (ca.mtimes(Rd.T, R) - ca.mtimes(R.T, Rd))
        return 0.5 * self.vee_map(e_R_mat)

    def angular_velocity_error(self, W, R, Rd, Wd):
        return W - ca.mtimes(R.T, ca.mtimes(Rd, Wd))

    def tracking_error(self, state, tracking_ref, attitude_ref):
        ex = state[0:3] - tracking_ref[0:3]
        ev = state[3:6] - tracking_ref[3:6]
        R = ca.reshape(state[6:15], 3, 3)
        W = state[15:18]
        Wd = tracking_ref[6:9]
        eR = self.attitude_error(R, attitude_ref)
        eW = self.angular_velocity_error(W, R, attitude_ref, Wd)
        return ca.vertcat(ex, ev, eR, eW)

    def dynamics_update(self, state, control, mass, J, g):
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

        W_rhs = -ca.cross(W, ca.mtimes(J, W)) + M
        W_dot = ca.solve(J, W_rhs)
        W_half = W + 0.5 * self.dt * W_dot
        R_next = ca.mtimes(R, self.cayley_transform(W_half * self.dt))
        W_half_rhs = -ca.cross(W_half, ca.mtimes(J, W_half)) + M
        W_dot_half = ca.solve(J, W_half_rhs)
        W_next = W_half + 0.5 * self.dt * W_dot_half

        return ca.vertcat(x_next, v_next, ca.reshape(R_next, 9, 1), W_next)

    def build_reference_horizon(self, env):
        tracking_ref = []
        attitude_ref = []
        base_idx = env.idx

        for k in range(self.Np):
            ref_idx = min(base_idx + k, env.xd.shape[1] - 1)
            xd = env.xd[:, ref_idx]
            vd = env.vd[:, ref_idx]
            tracking_ref.append(np.concatenate([xd, vd, np.zeros(3)]))
            attitude_ref.append(np.eye(3).reshape(-1, order='F'))

        return np.concatenate(tracking_ref), np.concatenate(attitude_ref)

    def build_solver(self):
        X = ca.SX.sym('X', self.x_dim, self.Np + 1)
        U = ca.SX.sym('U', self.u_dim, self.Np)

        x0_param = ca.SX.sym('x0', self.x_dim)
        tracking_ref_param = ca.SX.sym('tracking_ref', 9 * self.Np)
        attitude_ref_param = ca.SX.sym('attitude_ref', 9 * self.Np)
        mass_param = ca.SX.sym('mass')
        g_param = ca.SX.sym('g')
        J_param = ca.SX.sym('J', 3, 3)

        params = ca.vertcat(x0_param, tracking_ref_param, attitude_ref_param, mass_param,
                            g_param, ca.reshape(J_param, 9, 1),)

        obj = 0
        g = [X[:, 0] - x0_param]

        for k in range(self.Np):
            tracking_ref_k = tracking_ref_param[9*k : 9*(k+1)]
            attitude_ref_k = ca.reshape(attitude_ref_param[9*k : 9*(k+1)], 3, 3)
            uk = U[:, k]

            xi = self.tracking_error(X[:, k], tracking_ref_k, attitude_ref_k)
            obj += ca.mtimes([xi.T, self.Qy, xi])
            obj += ca.mtimes([uk.T, self.Qu, uk])

            x_next = self.dynamics_update(
                X[:, k], uk, mass_param, J_param, g_param)
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

        x_block = np.full(self.x_dim * (self.Np + 1), -np.inf)
        x_block_upper = np.full(self.x_dim * (self.Np + 1), np.inf)
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

    def solve_nmpc(self, x0, tracking_ref, attitude_ref, mass, J, g):
        params = np.concatenate([x0, tracking_ref, attitude_ref, np.array([mass, g]),
                                J.reshape(-1, order='F')])

        init_guess = self.build_initial_guess(x0, mass, g)
        sol = self.solver(x0=init_guess, lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg, p=params)

        w_opt = np.array(sol['x']).flatten()
        x_size = self.x_dim * (self.Np + 1)
        X_opt = w_opt[:x_size].reshape((self.x_dim, self.Np + 1), order='F')
        U_opt = w_opt[x_size:].reshape((self.u_dim, self.Np), order='F')
        self.last_solution = {'X': X_opt, 'U': U_opt}
        return X_opt, U_opt, float(sol['f'])

    def run(self, env, record: bool = True):
        # States and parameters
        mass = env.uav_dynamics.get_mass()
        J = env.uav_dynamics.get_inertia_matrix()
        g = env.uav_dynamics.get_gravitational_acceleration()
        x = env.uav_dynamics.get_position()
        v = env.uav_dynamics.get_velocity()
        R = env.uav_dynamics.get_rotmat()
        W = env.uav_dynamics.get_angular_velocity()

        # Solve nonlinear model predictive control problem
        x0 = self.pack_state(x, v, R, W)
        tracking_ref, attitude_ref = self.build_reference_horizon(env)
        X_opt, U_opt, total_cost = self.solve_nmpc(
            x0, tracking_ref, attitude_ref, mass, J, g)
        u = U_opt[:, 0]

        if record:
            # Record data for plotting
            ex = x - tracking_ref[0:3]
            ev = v - tracking_ref[3:6]
            Rd = attitude_ref[0:9].reshape((3, 3), order='F')
            Wd = tracking_ref[6:9]
            eW = W - R.T @ Rd @ Wd
            self.time_arr[self.idx] = self.idx * self.dt
            self.cost_arr[self.idx] = total_cost
            self.eW_arr[:, self.idx] = eW
            self.ex_arr[:, self.idx] = ex
            self.ev_arr[:, self.idx] = ev
            self.W_arr[:, self.idx] = W
            self.M_arr[:, self.idx] = u[1:4]
            self.f_arr[self.idx] = u[0]
            self.u_arr[:, self.idx] = u

            # Record data for plotting
            self.idx += 1

        return np.array([u[0], u[1], u[2], u[3]])

    def plot_graph(self):
        plt.figure("Control inputs")
        plt.subplot(4, 1, 1)
        plt.plot(self.time_arr, self.M_arr[0, :])
        plt.grid(True)
        plt.title("Control inputs")
        plt.xlabel("time [s]")
        plt.ylabel("M_x")
        plt.subplot(4, 1, 2)
        plt.plot(self.time_arr, self.M_arr[1, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("M_y")
        plt.subplot(4, 1, 3)
        plt.plot(self.time_arr, self.M_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("M_z")
        plt.subplot(4, 1, 4)
        plt.plot(self.time_arr, self.f_arr[:])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("f")

        plt.figure("Angular rate error (eW)")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, np.rad2deg(self.eW_arr[0, :]))
        plt.grid(True)
        plt.title("Angular rate error (eW)")
        plt.xlabel("time [s]")
        plt.ylabel("x [deg/s]")
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, np.rad2deg(self.eW_arr[1, :]))
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [deg/s]")
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, np.rad2deg(self.eW_arr[2, :]))
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("z [deg/s]")

        plt.figure("Position error")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, self.ex_arr[0, :])
        plt.grid(True)
        plt.title("Position error")
        plt.xlabel("time [s]")
        plt.ylabel("x [m]")
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, self.ex_arr[1, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [m]")
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, self.ex_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("z [m]")

        plt.figure("Velocity error")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, self.ev_arr[0, :])
        plt.grid(True)
        plt.title("Velocity error")
        plt.xlabel("time [s]")
        plt.ylabel("x [m/s]")
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, self.ev_arr[1, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [m/s]")
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, self.ev_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("z [m/s]")

        plt.figure("NMPC objective")
        plt.plot(self.time_arr, self.cost_arr)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("cost")
