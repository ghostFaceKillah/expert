"""
Helpers for scripts like run_atari.py.
"""

import os
import gym

from baselines import logger
from baselines.acktr.monitor import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, PenalizeDying, StepPenalty, LimitLength, make_state_restoring_atari
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def make_atari_env(env_id, num_env, seed, random_state_reset=False, dataflow_config=None,
                   limit_len=None, limit_penalty=None, death_penalty=None, step_penalty=None,
                   wrapper_kwargs=None, start_index=0, only_positive_rewards=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    wrapper_kwargs['is_monte'] = 'MontezumaRevenge' in env_id
    wrapper_kwargs['is_pong'] = 'Pong' in env_id
    wrapper_kwargs['only_positive_rewards'] = only_positive_rewards
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            if random_state_reset:
                env = make_state_restoring_atari(env_id, dataflow_config)
            else:
                env = make_atari(env_id)
            env.seed(seed + rank)
            if death_penalty is not None and death_penalty != 0:
                env = PenalizeDying(env, death_penalty)
            if step_penalty is not None and step_penalty != 0:
                env = StepPenalty(env, step_penalty)
            if limit_len is not None:
                assert limit_penalty is not None
                env = LimitLength(env, limit_len, timeout_penalty=limit_penalty)
            monitor_fname = logger.get_dir() and os.path.join(logger.get_dir(), str(rank))
            env = Monitor(env, monitor_fname, allow_early_resets=True)
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)
    vec_env = SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    return vec_env


def make_eval_env(env_id, dumpdir=None, wrapper_kwargs=None, seed=None):
    if wrapper_kwargs is None: wrapper_kwargs = {}
    wrapper_kwargs['is_monte'] = 'MontezumaRevenge' in env_id
    wrapper_kwargs['is_pong'] = 'Pong' in env_id
    env = make_atari(env_id)
    if seed is not None:
        env.seed(seed)

    env = LimitLength(env, 20000, timeout_penalty=0.0)
    env = gym.wrappers.Monitor(
        env, dumpdir, video_callable=lambda x: True, force=True
    )
    return wrap_deepmind(env, frame_stack=True, save_original_reward=True, **wrapper_kwargs)


def make_mujoco_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help="What environment to learn", type=str, default="MontezumaRevengeNoFrameskip-v4")
    parser.add_argument('--gamma', help="discount rate for discounting future returns", type=float, default=0.995)
    parser.add_argument('--frame_stack', help="how many last frames to take as state", type=int, default=4)
    parser.add_argument('--num_env', help="how many parallel envs to do", type=int, default=32)
    parser.add_argument('--nsteps', help="number of steps after which we boostrap by value function", type=int, default=20)
    parser.add_argument('--expert_nbatch', help="batch size of expert data", type=int, default=256)
    parser.add_argument('--vf_coef', help="coefficient in front of value function part in the loss function", type=float, default=0.5)
    parser.add_argument('--ent_coef', help="coefficient in front of entropy part in the loss function", type=float, default=0.001)
    parser.add_argument('--max_grad_norm', help="Maximum norm of gradient in optimization", type=float, default=0.5)
    parser.add_argument('--lr', help="Initial learning rate", type=float, default=0.125)
    parser.add_argument('--lrschedule', help="Learning rate schedule linear/constant", type=str, default="constant")
    parser.add_argument('--death_penalty', help="What is negative reward for dying?", type=float, default=0.0)
    parser.add_argument('--limit_penalty', help="What is negative reward for hitting timeout limit", type=float, default=0.0)
    parser.add_argument('--limit_len', help="How many steps is the actor allowed to take in one episode", type=int, default=20000)
    parser.add_argument('--step_penalty', help="What is negative reward for taking each step (sort of even harder discount)", type=float, default=0.0)
    parser.add_argument('--exp_adv_est', help="Expert advantage estimator", type=str, default="critic")
    parser.add_argument('--exp_max_score', help="Discard training transitions with total score far above this parameter,"
                                                " to bump up relative frequency of beginning-of-game data in early training.", type=float, default=30000)
    parser.add_argument('--exp_coeff', help="How to weight expert loss in the training loss function", type=float, default=0.125)
    parser.add_argument('--load_model', help="What .npy saved model weights to load", type=str, default=None)
    parser.add_argument('--random_state_reset', help="should we reset the environment to random expert state?", type=bool, default=False)
    parser.add_argument('--use_n_trajectories', help="How many trajectories should we use?", type=int, default=-1)

    return parser


def atari_eval_arg_parser():
    """
    Create an argparse.ArgumentParser for run_eval.py and run_video_writer.py
    """
    parser = arg_parser()
    parser.add_argument('--env', help="What environment to learn", type=str, default="MontezumaRevengeNoFrameskip-v4")
    parser.add_argument('--model', help='What model to load', type=str, required=True)
    parser.add_argument('--eval_type', help='What evaluation style to run - prob or argmax', default='prob')

    return parser
