"""
CPG in polar coordinates based on:
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""

# Registers PybulletEngine
import eagerx
import eagerx_pybullet  # noqa: F401

# Registers PybulletEngine
from eagerx_quadruped.helper import add_quadruped

# TODO: Use scipy to accurately integrate cpg
# todo: Specify realistic spaces for cpg.inputs.offset, cpg.outputs.xs_zs
# todo: Specify realistic spaces for quadruped.sensors.{base_orientation, base_velocity, base_position, vel, pos, force_torque}
# todo: Reduce dimension of force_torque sensor (gives [Fx, Fy, Fz, Mx, My, Mz] PER joint --> 6 * 12=72 dimensions).
# todo: Tune sensor rates to the lowest possible.

if __name__ == "__main__":

    T = 10.0
    env_rate = 20
    cpg_rate = 200
    cc_rate = 200
    quad_rate = 200
    sim_rate = 200
    # set_random_seed(1)

    roscore = eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.INFO)

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Add quadruped
    for i in range(2):
        add_quadruped(graph, f"quad_{i}", ["base_pos"], [0+i, 0, 0.33], base_orientation=[0, 0, 0, 1])

    engine = eagerx.Engine.make(
        "PybulletEngine",
        rate=sim_rate,
        gui=True,
        egl=True,
        sync=True,
        real_time_factor=0,
        process=eagerx.process.NEW_PROCESS,
    )

    # Define environment
    import gym
    from typing import Tuple, Dict


    class ComposedEnv(eagerx.BaseEnv):
        def __init__(self, name, rate, graph, engine, force_start=True):
            super(ComposedEnv, self).__init__(name, rate, graph, engine, force_start=force_start)
            self.steps = None

        @property
        def observation_space(self) -> gym.spaces.Dict:
            return self._observation_space

        @property
        def action_space(self) -> gym.spaces.Dict:
            return self._action_space

        def reset(self):
            # Reset number of steps
            self.steps = 0
            # Sample desired states
            states = self.state_space.sample()
            # Perform reset
            obs = self._reset(states)
            return obs

        def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
            # Apply action
            obs = self._step(action)
            self.steps += 1

            # Determine when is the episode over
            # currently just a timeout after 100 steps
            done = self.steps > int(T * env_rate)

            # Set info, tell the algorithm the termination was due to a timeout
            # (the episode was truncated)
            info = {"TimeLimit.truncated": self.steps > int(T * env_rate)}

            return obs, 0., done, info

    # Initialize Environment
    env = ComposedEnv(name="rx", rate=20, graph=graph, engine=engine)

    action = env.action_space.sample()
    while True:
        obs, done = env.reset(), False
        while not done:
            _, reward, done, info = env.step(action)
