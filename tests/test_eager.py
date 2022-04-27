# Registers butterworth_filter
import eagerx.nodes  # noqa: F401

# Registers PybulletBridge
import eagerx_pybullet  # noqa: F401

# Registers PybulletBridge
from eagerx.core.env import EagerxEnv
from eagerx.core.graph import Graph

import eagerx_quadruped.object  # noqa: F401
import eagerx_quadruped.robots.go1.configs_go1 as go1_config  # noqa: F401


def test_eagerx(skip=True):
    # disable test for now
    if skip:
        return
    roscore = eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.INFO)

    # Initialize empty graph
    graph = Graph.create()
    rate = 100  # in Hz

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
        done = steps > 200
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

    # obs_space = env.observation_space

    # Evaluate
    try:
        for eps in range(5):
            print(f"Episode {eps}")
            _, done = env.reset(), False
            while not done:
                action = env.action_space.sample()
                # action = dict(joints=go1_config.INIT_JOINT_ANGLES)
                obs, reward, done, info = env.step(action)
                # rgb = env.render("rgb_array")
    except KeyboardInterrupt:
        raise

    print("Shutting down")
    env.shutdown()
    if roscore:
        roscore.shutdown()
    print("Shutdown")


if __name__ == "__main__":
    test_eagerx(skip=False)
