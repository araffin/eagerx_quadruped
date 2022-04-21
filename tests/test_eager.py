# Registers butterworth_filter
import eagerx.nodes  # noqa: F401

# Registers PybulletBridge
import eagerx_pybullet  # noqa: F401

# Registers PybulletBridge
from eagerx.core.env import EagerxEnv
from eagerx.core.graph import Graph

import eagerx_quadruped.object  # noqa: F401


def test_eagerx():
    # disable test for now
    return
    roscore = eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.INFO)

    # Initialize empty graph
    graph = Graph.create()

    # Create solid object
    # cube = eagerx.Object.make("Solid", "cube", urdf="cube_small.urdf", rate=5.0)
    # graph.add(cube)

    # Create robot
    robot = eagerx.Object.make(
        "Quadruped",
        "quadruped",
        sensors=["pos"],
        actuators=["joint_control"],
        states=["pos"],
        rate=5.0,
        control_mode="position_control",
        self_collision=False,
    )
    graph.add(robot)

    # Connect the nodes
    graph.connect(action="joints", target=robot.actuators.joint_control)
    graph.connect(observation="position", source=robot.sensors.pos)

    # Show in the gui
    graph.gui()

    # Define bridgesif
    bridge = eagerx.Bridge.make(
        "PybulletBridge",
        rate=20.0,
        gui=True,
        egl=True,
        is_reactive=True,
        real_time_factor=0,
        process=eagerx.process.NEW_PROCESS,
    )

    # Define step function
    def step_fn(prev_obs, obs, action, steps):
        # Calculate reward
        rwd = 0
        # Determine done flag
        done = steps > 500
        # Set info:
        info = dict()
        return obs, rwd, done, info

    # Initialize Environment
    env = EagerxEnv(
        name="rx",
        rate=5.0,
        graph=graph,
        bridge=bridge,
        step_fn=step_fn,
        exclude=["at"],
    )

    # obs_space = env.observation_space

    # Evaluate
    # for eps in range(5000):
    #     print(f"Episode {eps}")
    #     _, done = env.reset(), False
    #     while not done:
    #         action = env.action_space.sample()
    #         obs, reward, done, info = env.step(action)
    #         # rgb = env.render("rgb_array")

    print("Shutting down")
    env.shutdown()
    if roscore:
        roscore.shutdown()
    print("Shutdown")
