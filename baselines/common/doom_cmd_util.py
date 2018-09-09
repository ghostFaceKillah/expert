import gym
import os

from baselines import logger
from baselines.bench import Monitor, GlobalMonitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import wrap_doom_deepmind_like, LimitLength
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from doom import env as doom_env


def make_doom_env(num_env, seed, channels, start_index=0, wrapper_kwargs=None, very_sparse=False):
    """ Create a wrapped, monitored Doom SubprocVecEnv. """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_env(rank):
        def _thunk():
            if very_sparse:
                env = doom_env.DoomMyWayHomeFixed15Env()
            else:
                env = doom_env.DoomMyWayHomeEnv()
            env.seed(seed + rank)
            monitor_fname = logger.get_dir() and os.path.join(logger.get_dir(), str(rank))
            env = Monitor(env, monitor_fname, rank)

            return wrap_doom_deepmind_like(env, **wrapper_kwargs)
        return _thunk

    set_global_seeds(seed)
    vec_env = SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    vec_env = GlobalMonitor(vec_env, list(range(num_env)), channels)
    return vec_env


def make_doom_eval_env(dumpdir=None, seed=None, very_sparse=False):
    wrapper_kwargs = {}
    if very_sparse:
        env = doom_env.DoomMyWayHomeFixed15Env()
    else:
        env = doom_env.DoomMyWayHomeEnv()
    if seed is not None:
        env.seed(seed)

    env = LimitLength(env, 20000, timeout_penalty=0.0)
    env = gym.wrappers.Monitor(
        env, dumpdir, video_callable=lambda x: True, force=True
    )

    env = wrap_doom_deepmind_like(env, frame_stack=True, save_original_reward=True, **wrapper_kwargs)
    env.reset()

    return env