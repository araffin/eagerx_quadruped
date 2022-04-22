import os
from typing import List, Optional

import eagerx.core.register as register
import numpy as np
from eagerx import EngineNode, EngineState, Object, SpaceConverter
from eagerx.core.graph_engine import EngineGraph
from eagerx.core.specs import ObjectSpec

# EAGERx IMPORTS
from eagerx_pybullet.bridge import PybulletBridge

# ROS IMPORTS
from std_msgs.msg import Float32MultiArray

import eagerx_quadruped.robots.go1.configs_go1 as go1_config


class Quadruped(Object):
    entity_id = "Quadruped"

    @staticmethod
    @register.sensors(pos=Float32MultiArray)
    @register.actuators(joint_control=Float32MultiArray)
    @register.engine_states(
        pos=Float32MultiArray,
        base_pos=Float32MultiArray,
        base_orientation=Float32MultiArray,
        base_velocity=Float32MultiArray,
        base_angular_velocity=Float32MultiArray,
    )
    @register.config(
        joint_names=None,
        gripper_names=None,
        fixed_base=False,
        self_collision=True,
        base_pos=None,
        base_or=None,
        control_mode=None,
    )
    def agnostic(spec: ObjectSpec, rate):
        """This methods builds the agnostic definition for a vx300s manipulator.

        Registered (agnostic) config parameters (should probably be set in the spec() function):
        - joint_names: List of arm joints.
        - gripper_names: List of gripper joints.
        - fixed_base: Force the base of the loaded object to be static.
        - self_collision: Enable self collisions.
        - base_pos: Base position of the object [x, y, z].
        - base_or: Base orientation of the object in quaternion [x, y, z, w].
        - control_mode: Control mode for the arm joints.
                        Available: `position_control`, `velocity_control`, `pd_control`, and `torque_control`.

        :param spec: Holds the desired configuration.
        :param rate: Rate (Hz) at which the callback is called.
        """
        # Register standard converters, space_converters, and processors
        import eagerx.converters  # noqa

        # Set observation properties: (space_converters, rate, etc...)
        spec.sensors.pos.rate = rate
        spec.sensors.pos.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=go1_config.RL_LOWER_ANGLE_JOINT.tolist(),
            high=go1_config.RL_UPPER_ANGLE_JOINT.tolist(),
        )

        # Set actuator properties: (space_converters, rate, etc...)
        spec.actuators.joint_control.rate = rate
        spec.actuators.joint_control.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=go1_config.RL_LOWER_ANGLE_JOINT.tolist(),
            high=go1_config.RL_UPPER_ANGLE_JOINT.tolist(),
        )

        # Set model_state properties: (space_converters)
        spec.states.pos.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=go1_config.INIT_JOINT_ANGLES.tolist(),
            high=go1_config.INIT_JOINT_ANGLES.tolist(),
        )

        spec.states.base_pos.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=go1_config.INIT_POSITION,
            high=go1_config.INIT_POSITION,
        )

        spec.states.base_orientation.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=list(go1_config.INIT_ORIENTATION),
            high=list(go1_config.INIT_ORIENTATION),
        )

        spec.states.base_velocity.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[0, 0, 0],
            high=[0, 0, 0],
        )

        spec.states.base_angular_velocity.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[0, 0, 0],
            high=[0, 0, 0],
        )

    @staticmethod
    @register.spec(entity_id, Object)
    def spec(
        spec: ObjectSpec,
        name: str,
        sensors: Optional[List[str]] = None,
        actuators: Optional[List[str]] = None,
        states: Optional[List[str]] = None,
        rate: float = 30.0,
        base_pos: Optional[List[int]] = None,
        base_or: Optional[List[int]] = None,
        self_collision: bool = False,
        fixed_base: bool = True,
        control_mode: str = "position_control",
    ):
        """A spec to create a go1 robot.

        :param spec: The desired object configuration.
        :param name: Name of the object (topics are placed within this namespace).
        :param sensors: A list of selected sensors. Must be a subset of the registered sensors.
        :param actuators: A list of selected actuators. Must be a subset of the registered actuators.
        :param states: A list of selected states. Must be a subset of the registered actuators.
        :param rate: The default rate at which all sensors and actuators run. Can be modified via the spec API.
        :param base_pos: Base position of the object [x, y, z].
        :param base_or: Base orientation of the object in quaternion [x, y, z, w].
        :param self_collision: Enable self collisions.
        :param fixed_base: Force the base of the loaded object to be static.
        :param control_mode: Control mode for the arm joints. Available: `position_control`, `velocity_control`, `pd_control`, and `torque_control`.
        :return: ObjectSpec
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        Quadruped.initialize_spec(spec)

        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = sensors if sensors else ["pos"]
        spec.config.actuators = actuators if actuators else ["joint_control"]
        spec.config.states = states if states else ["pos"]

        # Add registered agnostic params
        spec.config.joint_names = list(go1_config.JOINT_NAMES)
        spec.config.base_pos = base_pos if base_pos else go1_config.INIT_POSITION
        spec.config.base_or = base_or if base_or else [0, 0, 0, 1]
        spec.config.self_collision = self_collision
        spec.config.fixed_base = fixed_base
        spec.config.control_mode = control_mode

        # Add agnostic implementation
        Quadruped.agnostic(spec, rate)

    @staticmethod
    # This decorator pre-initializes bridge implementation with default object_params
    @register.bridge(entity_id, PybulletBridge)
    def pybullet_bridge(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Pybullet) of the object."""
        # Import any object specific entities for this bridge
        import eagerx_pybullet  # noqa

        # Set object arguments (as registered per register.bridge_params(..) above the bridge.add_object(...) method.
        urdf_file = os.path.join(go1_config.URDF_ROOT, go1_config.URDF_FILENAME)
        spec.PybulletBridge.urdf = urdf_file
        spec.PybulletBridge.basePosition = spec.config.base_pos
        spec.PybulletBridge.baseOrientation = spec.config.base_or
        spec.PybulletBridge.fixed_base = spec.config.fixed_base
        spec.PybulletBridge.self_collision = spec.config.self_collision

        # Create engine_states (no agnostic states defined in this case)
        spec.PybulletBridge.states.pos = EngineState.make("JointState", joints=spec.config.joint_names, mode="position")

        spec.PybulletBridge.states.base_pos = EngineState.make("LinkState", mode="position")
        spec.PybulletBridge.states.base_orientation = EngineState.make("LinkState", mode="orientation")
        spec.PybulletBridge.states.base_velocity = EngineState.make("LinkState", mode="velocity")
        spec.PybulletBridge.states.base_angular_velocity = EngineState.make("LinkState", mode="angular_vel")

        # Create sensor engine nodes
        # Rate=None, but we will connect them to sensors (thus will use the rate set in the agnostic specification)
        pos_sensor = EngineNode.make(
            "JointSensor", "pos_sensor", rate=spec.sensors.pos.rate, process=2, joints=spec.config.joint_names, mode="position"
        )

        # Create actuator engine nodes
        # Rate=None, but we will connect it to an actuator (thus will use the rate set in the agnostic specification)
        joint_control = EngineNode.make(
            "JointController",
            "joint_control",
            rate=spec.actuators.joint_control.rate,
            process=2,
            joints=spec.config.joint_names,
            mode=spec.config.control_mode,
            vel_target=np.zeros(len(go1_config.JOINT_NAMES)).tolist(),
            pos_gain=np.ones(len(go1_config.JOINT_NAMES)).tolist(),
            vel_gain=np.ones(len(go1_config.JOINT_NAMES)).tolist(),
        )

        # Connect all engine nodes
        graph.add([pos_sensor, joint_control])
        graph.connect(source=pos_sensor.outputs.obs, sensor="pos")
        graph.connect(actuator="joint_control", target=joint_control.inputs.action)

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)
