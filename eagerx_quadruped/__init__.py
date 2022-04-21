from gym.envs.registration import register

__version__ = "0.1.1"


register(
    id="QuadrupedSpring-v0",
    entry_point="eagerx_quadruped.gym_env.quadruped_gym_env:QuadrupedGymEnv",
    kwargs={
        "motor_control_mode": "CARTESIAN_PD",
        "task_env": "LR_COURSE_TASK",
        "observation_space_mode": "LR_COURSE_OBS",
    },
)
