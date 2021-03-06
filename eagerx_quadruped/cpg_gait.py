from typing import Dict, List, Optional

import eagerx
import numpy as np
from eagerx import register
from eagerx.utils.utils import Msg
from std_msgs.msg import Float32MultiArray

from eagerx_quadruped.hopf_network import HopfNetwork


class CpgGait(eagerx.Node):
    @staticmethod
    @register.spec("CpgGait", eagerx.Node)
    def spec(
        spec: eagerx.specs.NodeSpec,
        name: str,
        rate: float,
        gait: Dict,
        omega_swing: List[float],
        omega_stance: List[float],
        ground_clearance: float = 0.04,
        ground_penetration: float = 0.02,
        mu: int = 2,
        couple: bool = True,
        coupling_strength: float = 1.0,
        robot_height: float = 0.25,
        des_step_len: float = 0.04,
        process: Optional[int] = eagerx.process.NEW_PROCESS,
    ):
        """A spec to create a CpgGait node that produces a quadruped gait.

        It uses a CPG network based on hopf polar equations mapped to foot positions in Cartesian space.

        :param spec: Holds the desired configuration in a Spec object.
        :param name: User specified node name.
        :param rate: Rate (Hz) at which the callback is called.
        :param gait: Change depending on desired gait.
        :param omega_swing: todo: MUST EDIT
        :param omega_stance:  todo: MUST EDIT
        :param ground_clearance: Foot swing height.
        :param ground_penetration: Foot stance penetration into ground.
        :param mu: todo: 1**2, converge to sqrt(mu)
        :param couple: Should couple.
        :param coupling_strength: Coefficient to multiply coupling matrix.
        :param robot_height: In nominal case (standing).
        :param des_step_len: Desired step length.
        :param process: Process in which this node is launched. See :class:`~eagerx.core.constants.process` for all options.
        :return: NodeSpec
        """
        # Modify default params
        spec.config.update(name=name, rate=rate, process=process, inputs=["offset"], outputs=["cartesian_pos", "xs_zs"])

        # Modify params (args to .initialize())
        spec.config.update(mu=mu, gait=gait, omega_swing=omega_swing, omega_stance=omega_stance)
        spec.config.update(ground_clearance=ground_clearance, ground_penetration=ground_penetration)
        spec.config.update(couple=couple, coupling_strength=coupling_strength)
        spec.config.update(robot_height=robot_height, des_step_len=des_step_len)

        # TODO Define action limits
        # TODO: limit to 4 outputs instead of 12
        spec.inputs.offset.space_converter = eagerx.SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[-0.01] * 4,
            high=[0.01] * 4,
        )

        # Experimentally obtained. Above offset should be taken into account --> include y?
        spec.outputs.xs_zs.space_converter = eagerx.SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[-0.05656145, -0.26999995, -0.05656852, -0.2699973],  # expected high of unique [xs, zs, xs, zs] values
            high=[0.05636625, -0.21000053, 0.05642071, -0.21001561],  # expected high of unique [xs, zs, xs, zs] values
        )

    def initialize(
        self,
        mu,
        gait,
        omega_swing,
        omega_stance,
        ground_clearance,
        ground_penetration,
        couple,
        coupling_strength,
        robot_height,
        des_step_len,
    ):
        assert gait == "TROT", "xs_zs is only correct for TROT gait."
        self.n_legs = 4
        self.side_sign = np.array([-1, 1, -1, 1])  # get correct hip sign (body right is negative)
        self.foot_y = 0.0838  # this is the hip length
        self.cpg = HopfNetwork(
            mu=mu,
            gait=gait,
            omega_swing=omega_swing,
            omega_stance=omega_stance,
            time_step=0.005,  # Always update cpg with 200 Hz.
            ground_clearance=ground_clearance,  # foot swing height
            ground_penetration=ground_penetration,  # foot stance penetration into ground
            robot_height=robot_height,  # in nominal case (standing)
            des_step_len=des_step_len,  # 0 for jumping
        )

    @register.states()
    def reset(self):
        self.cpg.reset()

    @register.inputs(offset=Float32MultiArray)
    @register.outputs(cartesian_pos=Float32MultiArray, xs_zs=Float32MultiArray)
    def callback(self, t_n: float, offset: Msg):
        # update CPG
        while self.cpg.t <= t_n:
            self.cpg.update()

        # get desired foot positions from CPG
        xs, zs = self.cpg.get_xs_zs()

        # get unique xs & zs positions (BASED ON TROT)
        unique_xs_zs = np.array([xs[0], zs[0], xs[1], zs[1]], dtype="float32")

        action = np.zeros((12,))
        offset = offset.msgs[-1].data
        for i in range(self.n_legs):
            xyz_desired = np.array([xs[i], self.side_sign[i] * self.foot_y + offset[i], zs[i]])
            action[3 * i : 3 * i + 3] = xyz_desired

        # Add offset
        # TODO: clip
        # action += np.clip(np.array(offset.msgs[-1].data, dtype="float32"), -0.01, 0.01)
        # action += np.array(offset.msgs[-1].data, dtype="float32")
        return dict(cartesian_pos=Float32MultiArray(data=action), xs_zs=Float32MultiArray(data=unique_xs_zs))
