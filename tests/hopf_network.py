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
import pybullet
from eagerx.wrappers import Flatten
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

import eagerx_quadruped.cartesian_control  # noqa: F401
import eagerx_quadruped.cpg_gait  # noqa: F401

# Registers PybulletEngine
import eagerx_quadruped.object  # noqa: F401
import eagerx_quadruped.robots.go1.configs_go1 as go1_config  # noqa: F401

# from stable_baselines3.common.env_checker import check_env


# TODO: Use scipy to accurately integrate cpg
# todo: Specify realistic spaces for cpg.inputs.offset, cpg.outputs.xs_zs
# todo: Specify realistic spaces for quadruped.sensors.{base_orientation, base_velocity, base_position, vel, pos, force_torque}
# todo: Reduce dimension of force_torque sensor (gives [Fx, Fy, Fz, Mx, My, Mz] PER joint --> 6 * 12=72 dimensions).
# todo: Tune sensor rates to the lowest possible.

if __name__ == "__main__":

    episode_timeout = 10.0  # in s
    env_rate = 20
    cpg_rate = 200
    cartesian_rate = 200
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
    cartesian_control = eagerx.Node.make("CartesiandPDController", "cartesian_control", rate=cartesian_rate)
    graph.add(cartesian_control)

    # Create cpg node
    gait = "TROT"
    omega_swing, omega_stance = {
        "JUMP": [4 * np.pi, 40 * np.pi],
        "TROT": [16 * np.pi, 4 * np.pi],
        "WALK": [16 * np.pi, 4 * np.pi],
        "PACE": [20 * np.pi, 20 * np.pi],
        "BOUND": [10 * np.pi, 20 * np.pi],
    }[gait]
    cpg = eagerx.Node.make(
        "CpgGait",
        "cpg",
        rate=cpg_rate,
        gait=gait,
        omega_swing=omega_swing,
        omega_stance=omega_stance,
    )
    graph.add(cpg)

    # Connect the nodes
    graph.connect(action="offset", target=cpg.inputs.offset)
    graph.connect(source=cpg.outputs.cartesian_pos, target=cartesian_control.inputs.cartesian_pos)
    graph.connect(source=cartesian_control.outputs.joint_pos, target=robot.actuators.joint_control)
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
    # graph.gui()

    # Define engine
    engine = eagerx.Engine.make(
        "PybulletEngine",
        rate=sim_rate,
        gui=False,
        egl=True,
        sync=True,
        real_time_factor=0,
        process=eagerx.process.NEW_PROCESS,
    )

    # Define step function
    def step_fn(prev_obs, obs, action, steps):
        # Calculate reward
        # Go forward
        # desired_velocity = np.array([1.0, 0.0])
        # Go on the side, in circle
        desired_velocity = np.array([0.3, 0.4])
        alive_bonus = 1.0
        reward = alive_bonus - np.linalg.norm(desired_velocity - obs["base_vel"][0][:2])
        # print(obs["base_vel"][0][:2])
        # print(reward)
        # Convert Quaternion to Euler
        roll, pitch, yaw = pybullet.getEulerFromQuaternion(obs["base_orientation"][0])
        # print(list(map(np.rad2deg, (roll, pitch, yaw))))
        has_fallen = abs(np.rad2deg(roll)) > 40 or abs(np.rad2deg(pitch)) > 40
        timeout = steps > int(episode_timeout * env_rate)

        # Determine done flag
        done = timeout or has_fallen
        # Set info:
        info = {"TimeLimit.truncated": timeout}
        return obs, reward, done, info

    # Initialize Environment
    env = eagerx.EagerxEnv(name="rx", rate=20, graph=graph, engine=engine, step_fn=step_fn)
    env = Flatten(env)

    # model = TQC.load("logs/rl_model_30000_steps.zip")
    # mean_reward, std = evaluate_policy(model, env, n_eval_episodes=5)
    # print(f"Mean reward = {mean_reward:.2f} +/- {std}")
    # exit()

    # env = check_env(env)
    model = TQC(
        "MlpPolicy",
        env,
        learning_rate=7e-4,
        tau=0.02,
        gamma=0.98,
        buffer_size=300000,
        learning_starts=0,
        use_sde=True,
        use_sde_at_warmup=True,
        train_freq=8,
        gradient_steps=8,
        verbose=1,
        policy_kwargs=dict(n_critics=1),
    )

    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./logs/", name_prefix="rl_model")

    try:
        model.learn(1_000_000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        model.save("tqc_cpg")

    # while True:
    #     obs, done = env.reset(), False
    #     while not done:
    #
    #         action = np.zeros((12,))
    #         action[1] = -0.02
    #         action[4] = -0.02
    #         # action[7] = -0.02
    #         # action[10] = -0.02
    #         _, reward, done, info = env.step(action)
