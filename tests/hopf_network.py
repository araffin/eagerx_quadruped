"""
CPG in polar coordinates based on:
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""

# Registers PybulletEngine
import eagerx
import eagerx_pybullet  # noqa: F401
import numpy as np

# Registers PybulletEngine
import eagerx_quadruped.object  # noqa: F401
import eagerx_quadruped.cartesian_control  # noqa: F401
import eagerx_quadruped.cpg_gait  # noqa: F401
import eagerx_quadruped.robots.go1.configs_go1 as go1_config  # noqa: F401


# TODO: Use scipy to accurately integrate cpg
# todo: Implement sensors

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

    # Create robot
    robot = eagerx.Object.make(
        "Quadruped",
        "quadruped",
        actuators=["joint_control"],
        rate=quad_rate,
        control_mode="position_control",
        self_collision=False,
        fixed_base=False,
    )
    graph.add(robot)

    # Create cartesian control node
    cc = eagerx.Node.make("CartesiandPDController", "cartesian_control", rate=cc_rate)
    graph.add(cc)

    # Create cpg node
    gait = "TROT"
    omega_swing, omega_stance = {
        "JUMP": [4 * np.pi, 40 * np.pi],
        "TROT": [16 * np.pi, 4 * np.pi],
        "WALK": [16 * np.pi, 4 * np.pi],
        "PACE": [20 * np.pi, 20 * np.pi],
        "BOUND": [10 * np.pi, 20 * np.pi],
    }[gait]
    cpg = eagerx.Node.make("CpgGait", "cpg", rate=cpg_rate, gait=gait, omega_swing=omega_swing, omega_stance=omega_stance)
    graph.add(cpg)

    # Connect the nodes
    graph.connect(action="offset", target=cpg.inputs.offset)
    graph.connect(source=cpg.outputs.cartesian_pos, target=cc.inputs.cartesian_pos)
    graph.connect(source=cc.outputs.joint_pos, target=robot.actuators.joint_control)
    graph.connect(observation="position", source=robot.sensors.pos)

    # Show in the gui
    graph.gui()

    # Define engine
    engine = eagerx.Engine.make(
        "PybulletEngine",
        rate=sim_rate,
        gui=True,
        egl=True,
        sync=True,
        real_time_factor=0,
        process=eagerx.process.NEW_PROCESS,
    )

    # Define step function
    def step_fn(prev_obs, obs, action, steps):
        # Calculate reward
        rwd = 0
        # Determine done flag
        done = steps > int(T * env_rate)
        # Set info:
        info = dict()
        return obs, rwd, done, info

    # Initialize Environment
    env = eagerx.EagerxEnv(name="rx", rate=20, graph=graph, engine=engine, step_fn=step_fn)

    while True:
        obs, done = env.reset(), False
        while not done:
            action = np.zeros((12,))
            _, reward, done, info = env.step(dict(offset=action))
