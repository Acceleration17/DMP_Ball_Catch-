import numpy as np
import scipy.interpolate
import pickle


class DMP(object):
    """
        Dynamic Movement Primitives wlearned by Locally Weighted Regression (LWR).

    Implementation of P. Pastor, H. Hoffmann, T. Asfour and S. Schaal, "Learning and generalization of
    motor skills by learning from demonstration," 2009 IEEE International Conference on Robotics and
    Automation, 2009, pp. 763-768, doi: 10.1109/ROBOT.2009.5152385.
    """

    def __init__(self, nbasis=50, K_vec=10 * np.ones((6,)), weights=None):
        self.nbasis = nbasis  # Basis function number
        self.K_vec = K_vec

        self.K = np.diag(self.K_vec)  # Spring constant
        self.D = np.diag(2 * np.sqrt(self.K_vec))  # Damping constant, critically damped

        # used to determine the cutoff for s
        self.convergence_rate = 0.01
        self.alpha = -np.log(self.convergence_rate)

        # Creating basis functions and psi_matrix
        # Centers logarithmically distributed between 0.001 and 1
        self.basis_centers = np.logspace(-3, 0, num=self.nbasis)
        self.basis_variances = self.nbasis / (self.basis_centers**2)

        self.weights = weights

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                dict(
                    nbasis=self.nbasis,
                    K_vec=self.K_vec,
                    weights=self.weights,
                ),
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        dmp = cls(nbasis=data["nbasis"], K_vec=data["K_vec"], weights=data["weights"])
        return dmp

    @staticmethod
    def _gradient_nd(x, t):
        g1 = (x[:, 1:, :] - x[:, :-1, :]) / (t[:, 1:, None] - t[:, :-1, None])
        g = np.zeros_like(x)
        g[:, 0, :] = g1[:, 0, :]
        g[:, -1, :] = g1[:, -1, :]
        g[:, 1:-1, :] = (g1[:, 1:, :] + g1[:, :-1, :]) / 2
        return g

    def learn(self, X, T):
        """
        Learn the weights of the DMP using Locally Weighted Regression.

        X: demonstrated trajectories. Has shape [number of demos, number of timesteps, dofs].
        T: corresponding timings. Has shape [number of demos, number of timesteps].
            It is assumed that trajectories start at t=0
        """
        num_demos = X.shape[0]
        num_timesteps = X.shape[1]
        num_dofs = X.shape[2]

        x0 = X[:, 0, :]  # Shape: [num_demos, dofs]
        g = X[:, -1, :]  # Shape: [num_demos, dofs]
        tau = T[:, -1] - T[:, 0] + 1e-8  # Shape: [num_demos]

        if np.any(tau == 0):
            raise ValueError("Duration of the demonstration cannot be zero.")

        # Compute s(t) for each step in the demonstrations
        s = np.exp((-self.alpha / tau[:, None]) * T)  # Shape: [num_demos, num_timesteps]

        # Compute x_dot and x_ddot using numerical differentiation
        x_dot = self._gradient_nd(X, T)
        x_ddot = self._gradient_nd(x_dot, T)

        # Temporal Scaling by tau
        v = tau[:, None, None] * x_dot  # Shape: [num_demos, num_timesteps, num_dofs]
        v_dot = tau[:, None, None] * x_ddot  # Shape: [num_demos, num_timesteps, num_dofs]

        # Compute f_target(s) based on Equation 8 with scaling
        K_vec = self.K_vec[None, None, :]  # Shape: [1, 1, num_dofs]
        D_vec = (2 * np.sqrt(self.K_vec))[None, None, :]  # Shape: [1, 1, num_dofs]

        # Compute delta_g to handle varying goals
        delta_g = g - x0  # Shape: [num_demos, dofs]
        delta_g_norm = delta_g[:, None, :]  # Shape: [num_demos, 1, dofs]
        delta_g_norm[delta_g_norm == 0] = 1e-6  # Avoid division by zero

        # Compute f_target(s) with scaling
        f_s_target = ((v_dot + D_vec * v) / K_vec - (g[:, None, :] - X) + delta_g_norm * s[:, :, None]) / delta_g_norm

        # Compute psi(s)
        psi = np.exp(-self.basis_variances[None, None, :] * (s[:, :, None] - self.basis_centers[None, None, :]) ** 2)

        # Solve a least squares problem for the weights
        sum_psi = np.sum(psi, axis=2)  # Shape: [num_demos, num_timesteps]

        self.weights = np.zeros((num_dofs, self.nbasis))
        for d in range(num_dofs):
            # Prepare A and b for least squares
            A = (psi * s[:, :, None]).reshape(-1, self.nbasis)
            b = (f_s_target[:, :, d] * sum_psi).reshape(-1)
            # Solve for weights
            w_d, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            self.weights[d, :] = w_d

    def execute(self, t, dt, tau, x0, g, x_t, xdot_t):
        """
        Query the DMP at time t, with current position x_t, and velocity xdot_t.
        The parameter tau controls temporal scaling, x0 sets the initial position
        and g sets the goal for the trajectory.

        Returns the next position x_{t + dt} and velocity x_{t + dt}
        """
        if self.weights is None:
            raise ValueError("Cannot execute DMP before parameters are set by DMP.learn()")

        # Calculate s(t)
        s = np.exp(-self.alpha * t / tau)
        s = max(s, self.convergence_rate)  # Ensure s does not go below convergence rate

        # Compute psi(s)
        psi_s = np.exp(-self.basis_variances * (s - self.basis_centers) ** 2)  # Shape: [nbasis]
        sum_psi = np.sum(psi_s) + 1e-10  # Add small value to avoid division by zero

        # Compute delta_g to handle varying goals
        delta_g = g - x0
        delta_g[delta_g == 0] = 1e-6  # Avoid division by zero

        # Compute f(s) and scale it
        numerator = s * (psi_s @ self.weights.T)  # Shape: [dofs]
        f_s = (numerator / sum_psi) * delta_g  # Shape: [dofs]

        # Temporal Scaling
        v_t = tau * xdot_t

        # Calculate acceleration based on Equation 6 with scaling
        K_vec = self.K_vec  # Shape: [dofs]
        D_vec = 2 * np.sqrt(self.K_vec)  # Shape: [dofs]
        v_dot_t = K_vec * (g - x_t) - D_vec * v_t - K_vec * delta_g * s + K_vec * f_s  # Shape: [dofs]

        # Calculate next position and velocity
        v_tp1 = v_t + v_dot_t * dt
        xdot_tp1 = v_tp1 / tau
        x_tp1 = x_t + xdot_tp1 * dt

        return x_tp1, xdot_tp1

    def rollout(self, dt, tau, x0, g):
        time = 0
        x = x0
        x_dot = np.zeros_like(x0)
        X = [x0]

        while time <= tau:
            x, x_dot = self.execute(t=time, dt=dt, tau=tau, x0=x0, g=g, x_t=x, xdot_t=x_dot)
            time += dt
            X.append(x)

        return np.stack(X)

    def rollout2(self, time_steps, tau, x0, g):
        x = x0
        x_dot = np.zeros_like(x0)
        X = [x0]
        time_prev = time_steps[0]
        for t in time_steps[1:]:
            dt = t - time_prev
            x, x_dot = self.execute(t=t, dt=dt, tau=tau, x0=x0, g=g, x_t=x, xdot_t=x_dot)
            time_prev = t
            X.append(x)
        return np.stack(X)


    @staticmethod
    def interpolate(trajectories, initial_dt):
        """
        Combine the given variable length trajectories into a fixed length array
        by interpolating shorter arrays to the maximum given sequence length.

        trajectories: A list of N arrays of shape (T_i, num_dofs) where T_i is the number
            of time steps in trajectory i
        initial_dt: A scalar corresponding to the duration of each time step.
        """
        length = max(len(traj) for traj in trajectories)
        dofs = trajectories[0].shape[1]

        X = np.zeros((len(trajectories), length, dofs))
        T = np.zeros((len(trajectories), length))

        # TODO Your code goes here ...
        for i, traj in enumerate(trajectories):
            T_i = np.arange(traj.shape[0]) * initial_dt
            T_new = np.linspace(0, T_i[-1], length)
            interp_func = scipy.interpolate.interp1d(T_i, traj, axis=0, kind='linear')
            X[i] = interp_func(T_new)
            T[i] = T_new
        return X, T