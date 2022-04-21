import gym
import eagerx_quadruped

def test_gym_env():
    env = gym.make("QuadrupedSpring-v0", render=False)
    env.reset()
    for _ in range(100):
        env.step(env.action_space.sample())

    env.close()
