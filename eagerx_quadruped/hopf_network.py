import numpy as np
from scipy.integrate import odeint


class HopfNetwork:
    """CPG network based on hopf polar equations mapped to foot positions in Cartesian space.

    Foot Order is FR, FL, RR, RL
    (Front Right, Front Left, Rear Right, Rear Left)
    """

    def __init__(
        self,
        mu=1,  # 1**2,            # converge to sqrt(mu)
        omega_swing=1 * 2 * np.pi,  # MUST EDIT
        omega_stance=1 * 2 * np.pi,  # MUST EDIT
        gait="TROT",  # change depending on desired gait
        coupling_strength=1,  # coefficient to multiply coupling matrix
        couple=True,  # should couple
        time_step=0.001,  # time step
        ground_clearance=0.05,  # foot swing height
        ground_penetration=0.01,  # foot stance penetration into ground
        robot_height=0.25,  # in nominal case (standing)
        des_step_len=0.04,  # desired step length
    ):

        ###############
        # initialize CPG data structures: amplitude is row 0, and phase is row 1
        self.X = np.zeros((2, 4), dtype="float32")

        # save parameters
        self._mu = mu
        self._omega_swing = omega_swing
        self._omega_stance = omega_stance
        self._couple = couple
        self._coupling_strength = coupling_strength
        self._dt = time_step
        self._set_gait(gait)

        # save body and foot shaping
        self._ground_clearance = ground_clearance
        self._ground_penetration = ground_penetration
        self._robot_height = robot_height
        self._des_step_len = des_step_len

    def _set_gait(self, gait):
        """For coupling oscillators in phase space.
        [TODO] update all coupling matrices
        """
        self.PHI_trot = np.pi * np.array(
            [
                [0, -1, -1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, -1, -1, 0],
            ]
        )

        self.PHI_jump = np.pi * np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )

        self.PHI_walk = np.pi * np.array(
            [
                [0, -1, -1 / 2, 1 / 2],
                [1, 0, 1 / 2, 3 / 2],
                [1 / 2, -1 / 2, 0, 1],
                [-1 / 2, -3 / 2, -1, 0],
            ]
        )

        self.PHI_bound = np.pi * np.array(
            [
                [0, 0, -1, -1],
                [0, 0, -1, -1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ]
        )

        self.PHI_pace = np.pi * np.array(
            [
                [0, -1, 0, -1],
                [1, 0, 1, 0],
                [0, -1, 0, -1],
                [1, 0, 1, 0],
            ]
        )

        self.PHI = {
            "TROT": self.PHI_trot,
            "PACE": self.PHI_pace,
            "BOUND": self.PHI_bound,
            "WALK": self.PHI_walk,
            "JUMP": self.PHI_jump,
        }[gait]

        print(gait)

    def reset(self):
        # set oscillator initial conditions
        self.X[0, :] = 0.1
        # initialize so we are in stance phase for the trot gait
        self.X[1, :] = (self.PHI[0, :] + 4 / 3 * np.pi) % (2 * np.pi)

    def update(self):
        """Update oscillator states."""
        # todo: modify _integrate_hopf_equations to ode
        # x = self.X.reshape((-1,)).copy()
        # params = (self._omega_swing, self._omega_stance, self._mu, 50, self._couple, self._coupling_strength, self.PHI)
        # x = odeint(self.ode, x, [0., self._dt], args=params, rtol=2e-8, atol=2e-8, hmax=0.0, hmin=0.0, mxstep=0)[-1]
        # X = x.reshape(2, 4)
        # X[1, :] = X[1, :] % (2 * np.pi)
        # self.X = X

        # update parameters, integrate
        self._integrate_hopf_equations()

        # map CPG variables to Cartesian foot xz positions (Equations 8, 9)
        r = self.X[0, :]
        theta = self.X[1, :]
        x = -self._des_step_len * r * np.cos(theta)

        ground_clearance = self._ground_clearance * np.sin(theta)
        ground_penetration = self._ground_penetration * np.sin(theta)
        above_ground = np.sin(theta) > 0
        ground_offset = above_ground * ground_clearance + (1.0 - above_ground) * ground_penetration
        z = -self._robot_height + ground_offset

        return x, z

    def _integrate_hopf_equations(self):
        """Hopf polar equations and integration. Use equations 6 and 7."""
        # bookkeeping - save copies of current CPG states
        X = self.X.copy()
        X_dot = np.zeros((2, 4))
        alpha = 50

        # loop through each leg's oscillator
        for i in range(4):
            # get r_i, theta_i from X
            r = X[0, i]
            theta = X[1, i]
            # compute r_dot (Equation 6)
            r_dot = alpha * (self._mu - r**2) * r
            # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
            theta_dot = 0.0
            if np.sin(theta) > 0:
                theta_dot = self._omega_swing
            else:
                theta_dot = self._omega_stance

            # loop through other oscillators to add coupling (Equation 7)
            if self._couple:
                for j in range(4):
                    if j != i:
                        theta_dot += X[0, j] * self._coupling_strength * np.sin(X[1, j] - theta - self.PHI[i, j])

            # set X_dot[:,i]
            X_dot[:, i] = [r_dot, theta_dot]

        # integrate
        self.X += self._dt * X_dot
        # mod phase variables to keep between 0 and 2pi
        self.X[1, :] = self.X[1, :] % (2 * np.pi)

        # self.X_list.append(self.X.copy())
        # self.dX_list.append(X_dot)

    @staticmethod
    def ode(x, t, omega_swing, omega_stance, mu, alpha, couple, coupling_strength, PHI):
        r = x[:4]
        theta = x[4:]

        # Calculate r_dot
        dr = alpha * r * (mu - r**2)

        # Calculate theta dot
        dtheta = np.zeros(4, dtype="float32")
        mask = np.sin(theta) > 0
        dtheta[mask] = omega_swing
        dtheta[~mask] = omega_stance

        # Calculate coupling
        if couple:
            for i in range(4):
                for j in range(4):
                    if j != i:
                        dtheta[i] += r[j] * coupling_strength * np.sin(theta[j] - theta[i] - PHI[i, j])
        return dr.tolist() + dtheta.tolist()
