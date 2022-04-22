from typing import List, Optional

import eagerx.core.register as register
import pybullet
from eagerx.core.constants import process as p
from eagerx.core.entities import EngineNode
from eagerx.core.specs import NodeSpec
from eagerx.utils.utils import Msg
from std_msgs.msg import Float32MultiArray, UInt64


class JointController(EngineNode):
    @staticmethod
    @register.spec("JointController", EngineNode)
    def spec(
        spec: NodeSpec,
        name: str,
        rate: float,
        joints: List[str],
        process: Optional[int] = p.BRIDGE,
        color: Optional[str] = "green",
        mode: str = "position_control",
        vel_target: Optional[List[float]] = None,
        pos_gain: Optional[List[float]] = None,
        vel_gain: Optional[List[float]] = None,
        max_force: Optional[List[float]] = None,
    ):
        """A spec to create a JointController node that controls a set of joints.

        For more info on `vel_target`, `pos_gain`, and `vel_gain`, see `setJointMotorControlMultiDofArray` in
        https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#

        :param spec: Holds the desired configuration in a Spec object.
        :param name: User specified node name.
        :param rate: Rate (Hz) at which the callback is called.
        :param joints: List of controlled joints. Its order determines the ordering of the applied commands.
        :param process: Process in which this node is launched. See :class:`~eagerx.core.constants.process` for all options.
        :param color: Specifies the color of logged messages & node color in the GUI.
        :param mode: Available: `position_control`, `velocity_control`, `pd_control`, and `torque_control`.
        :param vel_target: The desired velocity. Ordering according to `joints`.
        :param pos_gain: Position gain. Ordering according to `joints`.
        :param vel_gain: Velocity gain. Ordering according to `joints`.
        :param max_force: Maximum force when mode in [`position_control`, `velocity_control`, `pd_control`]. Ordering
                          according to `joints`.
        :return: NodeSpec
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(JointController)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick", "action"]
        spec.config.outputs = ["action_applied"]

        # Set parameters, defined by the signature of cls.initialize(...)
        spec.config.joints = joints
        spec.config.mode = mode
        spec.config.vel_target = vel_target if vel_target else [0.0] * len(joints)
        spec.config.pos_gain = pos_gain if pos_gain else [0.2] * len(joints)
        spec.config.vel_gain = vel_gain if vel_gain else [0.2] * len(joints)
        spec.config.max_force = max_force if max_force else [999.0] * len(joints)

    def initialize(self, joints, mode, vel_target, pos_gain, vel_gain, max_force):
        # We will probably use self.simulator[self.obj_name] in callback & reset.
        self.obj_name = self.config["name"]
        assert self.process == p.BRIDGE, (
            "Simulation node requires a reference to the simulator," " hence it must be launched in the Bridge process"
        )
        flag = self.obj_name in self.simulator["robots"]
        assert flag, f'Simulator object "{self.simulator}" is not compatible with this simulation node.'
        self.joints = joints
        self.mode = mode
        self.vel_target = vel_target
        self.pos_gain = pos_gain
        self.vel_gain = vel_gain
        self.max_force = max_force
        self.robot = self.simulator["robots"][self.obj_name]
        self._p = self.simulator["client"]
        self.physics_client_id = self._p._client

        self.bodyUniqueId = []
        self.jointIndices = []
        for _idx, pb_name in enumerate(joints):
            bodyid, jointindex = self.robot.jdict[pb_name].get_bodyid_jointindex()
            self.bodyUniqueId.append(bodyid), self.jointIndices.append(jointindex)

        self.joint_cb = self._joint_control(
            self._p,
            self.mode,
            self.bodyUniqueId[0],
            self.jointIndices,
            self.pos_gain,
            self.vel_gain,
            self.vel_target,
            self.max_force,
        )

    @register.states()
    def reset(self):
        pass
        # self.simulator[self.obj_name]["input"] = np.squeeze(np.array(self.default_action))

    @register.inputs(tick=UInt64, action=Float32MultiArray)
    @register.outputs(action_applied=Float32MultiArray)
    def callback(
        self,
        t_n: float,
        tick: Optional[Msg] = None,
        action: Optional[Msg] = None,
    ):
        assert action is not None
        # Set action in pybullet
        self.joint_cb(action.msgs[-1].data)
        # Send action that has been applied.
        return dict(action_applied=action.msgs[-1])

    @staticmethod
    def _joint_control(p, mode, bodyUniqueId, jointIndices, pos_gain, vel_gain, vel_target, max_force):
        if mode == "position_control":

            def cb(action):
                return p.setJointMotorControlArray(
                    bodyUniqueId=bodyUniqueId,
                    jointIndices=jointIndices,
                    controlMode=pybullet.POSITION_CONTROL,
                    targetPositions=action,
                    targetVelocities=vel_target,
                    positionGains=pos_gain,
                    velocityGains=vel_gain,
                    forces=max_force,
                    physicsClientId=p._client,
                )

        elif mode == "velocity_control":

            def cb(action):
                return p.setJointMotorControlArray(
                    bodyUniqueId=bodyUniqueId,
                    jointIndices=jointIndices,
                    controlMode=pybullet.VELOCITY_CONTROL,
                    targetVelocities=action,
                    positionGains=pos_gain,
                    velocityGains=vel_gain,
                    forces=max_force,
                    physicsClientId=p._client,
                )

        elif mode == "pd_control":

            def cb(action):
                return p.setJointMotorControlArray(
                    bodyUniqueId=bodyUniqueId,
                    jointIndices=jointIndices,
                    controlMode=pybullet.PD_CONTROL,
                    targetVelocities=action,
                    positionGains=pos_gain,
                    velocityGains=vel_gain,
                    forces=max_force,
                    physicsClientId=p._client,
                )

        elif mode == "torque_control":

            def cb(action):
                return p.setJointMotorControlArray(
                    bodyUniqueId=bodyUniqueId,
                    jointIndices=jointIndices,
                    controlMode=pybullet.TORQUE_CONTROL,
                    forces=action,
                    positionGains=pos_gain,
                    velocityGains=vel_gain,
                    physicsClientId=p._client,
                )

        else:
            raise ValueError(f"Mode '{mode}' not recognized.")
        return cb
