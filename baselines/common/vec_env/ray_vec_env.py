import numpy as np
import ray

from baselines.common.vec_env import VecEnv


"""
envs = [
    GymEnvironment.remote("Pong-v0")
    for _ in range(10)
]

actions = [env.step.remote(0) for env in envs]

print("=" * 80)
print(ray.get(actions))
print("=" * 80)
"""


@ray.remote
class EnvActor(object):
    def __init__(self, env_creation_function):
        self.env = env_creation_function()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        if done:
            ob = self.env.reset()
        return ob, reward, done, info

    def reset(self):
        return self.env.reset()

    def reset_task(self):
        return self.env.reset_task()

    def get_space(self):
        return self.env.observation_space, self.env.action_space

    def close(self):
        self.env.close()
        return 1


class RayVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        env_fns: List of functions that create gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)

        self.actors = [EnvActor.remote(fn) for fn in env_fns]

        observation_space, action_space = ray.get(self.actors[0].get_space.remote())
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        self.step_futures = None

    def step_async(self, actions):
        step_futures = []

        for actor, action in zip(self.actors, actions):
            step_future = actor.step.remote(action)
            step_futures.append(step_future)

        self.waiting = True
        self.step_futures = step_futures

    def step_wait(self):
        results = ray.get(self.step_futures)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        acc = []
        for actor in self.actors:
            resu = actor.reset.remote()
            acc.append(resu)

        return np.stack(ray.get(acc))

    def reset_task(self):
        acc = []
        for actor in self.actors:
            resu = actor.reset_task.remote()
            acc.append(resu)

        return np.stack(ray.get(acc))

    def close(self):
        if self.closed:
            return
        if self.waiting:
            ray.get(self.step_futures)

        acc = []
        for actor in self.actors:
            resu = actor.close.remote()
            acc.append(resu)

        assert sum(ray.get(acc)) == self.nenvs



## Below imports are for the integration tests only
from baselines.common.atari_wrappers import make_atari, wrap_deepmind


def integration_test(env_id, num_env, seed):

    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            return wrap_deepmind(env, frame_stack=True)
        return _thunk

    env = RayVecEnv([make_env(i) for i in range(num_env)])
    return env


if __name__ == '__main__':
    ray.init()
    env = integration_test('PongNoFrameskip-v4', 2, 0)

