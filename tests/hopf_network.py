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

    # Create robot
    robot = eagerx.Object.make(
        "Quadruped",
        "quadruped",
        actuators=["joint_control"],
        # sensors=["pos", "vel", "force_torque", "base_orientation", "base_pos", "base_vel"],
        sensors=["pos", "vel", "base_orientation", "base_pos", "base_vel"],
        rate=quad_rate,
        control_mode="position_control",
        self_collision=False,
        fixed_base=False,
    )
    # TODO: tune sensor rates to the lowest possible.
    robot.sensors.pos.rate = env_rate
    robot.sensors.vel.rate = env_rate
    robot.sensors.force_torque.rate = env_rate
    robot.sensors.base_orientation.rate = env_rate
    robot.sensors.base_pos.rate = env_rate
    robot.sensors.base_vel.rate = env_rate
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
    graph.connect(observation="velocity", source=robot.sensors.vel)
    graph.connect(observation="base_pos", source=robot.sensors.base_pos)
    graph.connect(observation="base_vel", source=robot.sensors.base_vel)
    graph.connect(observation="base_orientation", source=robot.sensors.base_orientation)
    graph.connect(
        observation="xs_zs",
        source=cpg.outputs.xs_zs,
        skip=True,
        initial_obs=[-0.01354526, -0.26941818, 0.0552178, -0.25434446],
    )

    # Optionally, add force_torque sensor
    graph.add_component(robot.sensors.force_torque)
    graph.connect(observation="force_torque", source=robot.sensors.force_torque)

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
