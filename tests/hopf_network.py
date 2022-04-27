"""
CPG in polar coordinates based on:
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
# Registers butterworth_filter
import eagerx.nodes  # noqa: F401

# Registers PybulletBridge
import eagerx_pybullet  # noqa: F401
import numpy as np

# Registers PybulletBridge
from eagerx.core.env import EagerxEnv
from eagerx.core.graph import Graph
from stable_baselines3.common.utils import set_random_seed

import eagerx_quadruped.object  # noqa: F401
import eagerx_quadruped.robots.go1.configs_go1 as go1_config  # noqa: F401


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
        self.X = np.zeros((2, 4))

        # save parameters
        self._mu = mu
        self._omega_swing = omega_swing
        self._omega_stance = omega_stance
        self._couple = couple
        self._coupling_strength = coupling_strength
        self._dt = time_step
        self._set_gait(gait)

        # set oscillator initial conditions
        self.X[0, :] = 0.1
        # initialize so we are in stance phase for the trot gait
        self.X[1, :] = (self.PHI[0, :] + 4 / 3 * np.pi) % (2 * np.pi)

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

    def update(self):
        """Update oscillator states."""

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


if __name__ == "__main__":

    # TIME_STEP = 0.005
    # rate = 1 / TIME_STEP
    rate = 200  # in Hz
    TIME_STEP = 1 / rate
    foot_y = 0.0838  # this is the hip length
    side_sign = np.array([-1, 1, -1, 1])  # get correct hip sign (body right is negative)
    set_random_seed(1)

    roscore = eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.INFO)

    # Initialize empty graph
    graph = Graph.create()

    # Create robot
    robot = eagerx.Object.make(
        "Quadruped",
        "quadruped",
        actuators=["cartesian_control"],
        rate=rate,
        control_mode="position_control",
        self_collision=True,
        fixed_base=False,
    )
    graph.add(robot)

    # Connect the nodes
    # graph.connect(action="joints", target=robot.actuators.joint_control)
    graph.connect(action="cartesian_pos", target=robot.actuators.cartesian_control)
    graph.connect(observation="position", source=robot.sensors.pos)

    # Show in the gui
    # graph.gui()

    # Define bridgesif
    bridge = eagerx.Bridge.make(
        "PybulletBridge",
        rate=rate,
        gui=True,
        egl=True,
        sync=True,
        real_time_factor=0,
        process=eagerx.process.ENVIRONMENT,
    )

    # Define step function
    def step_fn(prev_obs, obs, action, steps):
        # Calculate reward
        rwd = 0
        # Determine done flag
        done = steps > 2000
        # Set info:
        info = dict()
        return obs, rwd, done, info

    # Initialize Environment
    env = EagerxEnv(
        name="rx",
        rate=rate,
        graph=graph,
        bridge=bridge,
        step_fn=step_fn,
    )

    env.reset()

    gait = "TROT"
    # initialize Hopf Network, supply gait
    omega_swing, omega_stance = {
        "JUMP": [4 * np.pi, 40 * np.pi],
        "TROT": [16 * np.pi, 4 * np.pi],
        "WALK": [16 * np.pi, 4 * np.pi],
        "PACE": [20 * np.pi, 20 * np.pi],
        "BOUND": [10 * np.pi, 20 * np.pi],
    }[gait]

    cpg = HopfNetwork(
        mu=2,
        gait=gait,
        omega_swing=omega_swing,
        omega_stance=omega_stance,
        time_step=TIME_STEP,
        ground_clearance=0.04,  # foot swing height
        ground_penetration=0.02,  # foot stance penetration into ground
        # robot_height=0.25,  # in nominal case (standing)
        # des_step_len = 0 for jumping
        # des_step_len=0.00,  # desired step length
    )

    T = 10.0
    TEST_STEPS = int(T / (TIME_STEP))

    for _ in range(TEST_STEPS):
        # get desired foot positions from CPG
        xs, zs = cpg.update()

        action = np.zeros((12,))
        n_legs = 4
        # loop through desired foot positions and calculate torques
        for i in range(n_legs):
            xyz_desired = np.array([xs[i], side_sign[i] * foot_y, zs[i]])
            # Set tau for legi in action vector
            action[3 * i : 3 * i + 3] = xyz_desired

        # send torques to robot and simulate TIME_STEP seconds
        action = dict(cartesian_pos=action)
        env.step(action)
