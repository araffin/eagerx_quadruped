from typing import List, Optional

import eagerx.core.register as register
import numpy as np
import pybullet
from eagerx.core.constants import process as p
from eagerx.core.entities import EngineNode
from eagerx.core.specs import NodeSpec
from eagerx.utils.utils import Msg
from std_msgs.msg import Float32MultiArray, UInt64

import eagerx_quadruped.robots.go1.configs_go1 as go1_config


class CartesiandPDController(EngineNode):
    @staticmethod
    @register.spec("CartesiandPDController", EngineNode)
    def spec(
        spec: NodeSpec,
        name: str,
        rate: float,
        joints: List[str],
        process: Optional[int] = p.BRIDGE,
        color: Optional[str] = "green",
        # mode: str = "position_control",
        vel_gain: Optional[List[float]] = None,
        vel_target: Optional[List[float]] = None,
        pos_gain: Optional[List[float]] = None,
        max_force: Optional[List[float]] = None,
    ):
        """A spec to create a CartesiandPDController node that controls a set of joints.

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
        :param max_force: Maximum force when mode in [`position_control`, `velocity_control`, `pd_control`]. Ordering
                          according to `joints`.
        :return: NodeSpec
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(CartesiandPDController)

        # Modify default node params
        spec.config.update(
            dict(
                name=name,
                rate=rate,
                process=process,
                inputs=["tick", "cartesian_pos"],
                outputs=["action_applied"],
                joints=joints,
            )
        )
        # spec.config.name = name
        # spec.config.rate = rate
        # spec.config.process = process
        # spec.config.inputs = ["tick", "cartesian_pos"]
        # spec.config.outputs = ["action_applied"]

        # Set parameters, defined by the signature of cls.initialize(...)
        # spec.config.joints = joints

        # spec.config.mode = mode
        spec.config.vel_target = vel_target if vel_target else [0.0] * len(joints)
        spec.config.pos_gain = pos_gain if pos_gain else [0.2] * len(joints)
        spec.config.vel_gain = vel_gain if vel_gain else [0.2] * len(joints)
        spec.config.max_force = max_force if max_force else [20.0] * len(joints)

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

        self.body_unique_id = []
        self.joint_indices = []
        for pb_name in joints:
            body_id, joint_id = self.robot.jdict[pb_name].get_bodyid_jointindex()
            self.body_unique_id.append(body_id), self.joint_indices.append(joint_id)

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=UInt64, cartesian_pos=Float32MultiArray)
    @register.outputs(action_applied=Float32MultiArray)
    def callback(
        self,
        t_n: float,
        tick: Msg,
        cartesian_pos: Msg,
    ):
        assert cartesian_pos is not None

        # desired [x, y, z] for each joint
        action = np.array(cartesian_pos.msgs[-1].data)
        desired_joint_angles = np.zeros(
            (
                len(
                    self.joint_indices,
                )
            )
        )
        # call inverse kinematics to get corresponding joint angles
        for leg_idx in range(go1_config.NUM_LEGS):
            xyz_desired = action[3 * leg_idx : 3 * (leg_idx + 1)]
            leg_q = self.compute_inverse_kinematics(leg_idx, xyz_desired)
            desired_joint_angles[3 * leg_idx : 3 * (leg_idx + 1)] = leg_q

        # Set action in pybullet
        self._p.setJointMotorControlArray(
            bodyUniqueId=self.body_unique_id[0],
            jointIndices=self.joint_indices,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=desired_joint_angles,
            targetVelocities=self.vel_target,
            positionGains=self.pos_gain,
            velocityGains=self.vel_gain,
            forces=self.max_force,
            physicsClientId=self.physics_client_id,
        )

        # Send action that has been applied.
        return dict(action_applied=desired_joint_angles)

    @staticmethod
    def compute_inverse_kinematics(leg_id: int, xyz_coord: np.ndarray):
        """Get joint angles for leg leg_id with desired xyz position in leg frame.

        Leg 0: FR; Leg 1: FL; Leg 2: RR ; Leg 3: RL;

        From SpotMicro:
        https://github.com/OpenQuadruped/spot_mini_mini/blob/spot/spotmicro/Kinematics/LegKinematics.py
        """
        # rename links
        shoulder_length = go1_config.HIP_LINK_LENGTH
        elbow_length = go1_config.THIGH_LINK_LENGTH
        wrist_length = go1_config.CALF_LINK_LENGTH
        # coords
        x = xyz_coord[0]
        y = xyz_coord[1]
        z = xyz_coord[2]

        # get_domain
        D = (y**2 + (-z) ** 2 - shoulder_length**2 + (-x) ** 2 - elbow_length**2 - wrist_length**2) / (
            2 * wrist_length * elbow_length
        )

        D = np.clip(D, -1.0, 1.0)

        # check Right vs Left leg for hip angle
        sideSign = 1
        if leg_id == 0 or leg_id == 2:
            sideSign = -1

        # Right Leg Inverse Kinematics Solver
        wrist_angle = np.arctan2(-np.sqrt(1 - D**2), D)
        sqrt_component = y**2 + (-z) ** 2 - shoulder_length**2
        if sqrt_component < 0.0:
            sqrt_component = 0.0
        shoulder_angle = -np.arctan2(z, y) - np.arctan2(np.sqrt(sqrt_component), sideSign * shoulder_length)
        elbow_angle = np.arctan2(-x, np.sqrt(sqrt_component)) - np.arctan2(
            wrist_length * np.sin(wrist_angle), elbow_length + wrist_length * np.cos(wrist_angle)
        )
        joint_angles = np.array([-shoulder_angle, elbow_angle, wrist_angle])
        return joint_angles

    # def scale_action_to_cartesian_pos(self, actions):
    #     """Scale RL action to Cartesian PD ranges.
    #     Edit ranges, limits etc., but make sure to use Cartesian PD to compute the torques.
    #     """
    #     # clip RL actions to be between -1 and 1 (standard RL technique)
    #     # print(actions)
    #     u = np.clip(actions, -1, 1)
    #     # scale to corresponding desired foot positions (i.e. ranges in x,y,z we allow the agent to choose foot positions)
    #     scale_array = np.array([0.2, 0.05, 0.15] * 4)  # [0.1, 0.05, 0.08]*4)
    #     # add to nominal foot position in leg frame (what are the final ranges?)
    #     des_foot_pos = go1_config.NOMINAL_FOOT_POS_LEG_FRAME + scale_array * u
    #     # get Cartesian kp and kd gains (can be modified)
    #     kpCartesian = go1_config.kpCartesian
    #     kdCartesian = go1_config.kdCartesian
    #     # get current motor velocities
    #     dq = self.robot.GetMotorVelocities()
    #
    #     action = np.zeros(12)
    #     for i in range(4):
    #         # Get current Jacobian and foot position in leg frame
    #         J, xyz = self.robot.ComputeJacobianAndPosition(i)
    #         # Get current foot velocity in leg frame
    #         dxyz = J @ dq[3 * i : 3 * (i + 1)]
    #         delta_foot_pos = xyz - des_foot_pos[3 * i : 3 * (i + 1)]
    #
    #         # clamp the motor command by the joint limit, in case weired things happens
    #         if self._enable_action_clipping:
    #             delta_foot_pos = np.clip(
    #                 delta_foot_pos,
    #                 -go1_config.MAX_CARTESIAN_FOOT_POS_CHANGE_PER_STEP,
    #                 go1_config.MAX_CARTESIAN_FOOT_POS_CHANGE_PER_STEP,
    #             )
    #
    #         F_foot = -kpCartesian @ delta_foot_pos - kdCartesian @ dxyz
    #         # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
    #         tau = J.T @ F_foot
    #         action[3 * i : 3 * (i + 1)] = tau  # TODO: add white noise
    #     return action
