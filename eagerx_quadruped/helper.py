import eagerx
import numpy as np

# Registers PybulletEngine
import eagerx_quadruped.object  # noqa: F401
import eagerx_quadruped.cartesian_control  # noqa: F401
import eagerx_quadruped.cpg_gait  # noqa: F401
import eagerx_quadruped.robots.go1.configs_go1 as go1_config  # noqa: F401


def add_quadruped(graph, name="quadruped", sensors=["base_pos", "base_vel"], base_pos=None, base_orientation=None, quad_rate=200, cc_rate=200, cpg_rate=200, sensor_rate=20):
    # Registers PybulletEngine
    import eagerx_quadruped.object  # noqa: F401
    import eagerx_quadruped.cartesian_control  # noqa: F401
    import eagerx_quadruped.cpg_gait  # noqa: F401
    import eagerx_quadruped.robots.go1.configs_go1 as go1_config  # noqa: F401

    # assert len(sensors) > 0, "at least 1 sensor must be selected."

    # Create robot
    robot = eagerx.Object.make(
        "Quadruped",
        name,
        actuators=["joint_control"],
        sensors=[],
        rate=quad_rate,
        control_mode="position_control",
        base_pos=base_pos,
        base_orientation=base_orientation,
        self_collision=False,
        fixed_base=False,
    )
    robot.sensors.pos.rate = sensor_rate
    robot.sensors.vel.rate = sensor_rate
    robot.sensors.force_torque.rate = sensor_rate
    robot.sensors.base_orientation.rate = sensor_rate
    robot.sensors.base_pos.rate = sensor_rate
    robot.sensors.base_vel.rate = sensor_rate
    graph.add(robot)

    # Create cartesian control node
    cc = eagerx.Node.make("CartesiandPDController", f"{name}_cc", rate=cc_rate)
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
    cpg = eagerx.Node.make("CpgGait", f"{name}_cpg", rate=cpg_rate, gait=gait, omega_swing=omega_swing, omega_stance=omega_stance)
    graph.add(cpg)

    # Connect the nodes
    graph.connect(action=f"{name}_offset", target=cpg.inputs.offset)
    graph.connect(source=cpg.outputs.cartesian_pos, target=cc.inputs.cartesian_pos)
    graph.connect(source=cc.outputs.joint_pos, target=robot.actuators.joint_control)

    if "position" in sensors:
        graph.add_component(robot.sensors.pos)
        graph.connect(observation=f"{name}_pos", source=robot.sensors.pos)
    if "velocity" in sensors:
        graph.add_component(robot.sensors.vel)
        graph.connect(observation=f"{name}_vel", source=robot.sensors.vel)
    if "base_pos" in sensors:
        graph.add_component(robot.sensors.base_pos)
        graph.connect(observation=f"{name}_base_pos", source=robot.sensors.base_pos)
    if "base_vel" in sensors:
        graph.add_component(robot.sensors.base_vel)
        graph.connect(observation=f"{name}_base_vel", source=robot.sensors.base_vel)
    if "base_orientation" in sensors:
        graph.add_component(robot.sensors.base_orientation)
        graph.connect(observation=f"{name}_base_orn", source=robot.sensors.base_orientation)
    if "force_torque" in sensors:
        graph.add_component(robot.sensors.force_torque)
        graph.connect(observation=f"{name}_ft", source=robot.sensors.force_torque)
    if "xs_zs" in sensors:
        graph.connect(
            observation=f"{name}_xs_zs",
            source=cpg.outputs.xs_zs,
            skip=True,
            initial_obs=[-0.01354526, -0.26941818, 0.0552178, -0.25434446],
        )
    return robot