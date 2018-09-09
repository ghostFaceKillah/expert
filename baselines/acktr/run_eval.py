#!/usr/bin/env python3

import os
import gym
import tensorflow as tf

from baselines import logger
from baselines.common.cmd_util import make_eval_env, atari_eval_arg_parser
from baselines.video.video_runners import Model
from baselines.acktr.policies import CnnPolicy

import constants as cnst
import helpers


def main(args):
    logger.configure(dir=cnst.openai_logdir())

    policy_fn = CnnPolicy
    env_name = args.env
    model_path = os.path.join(cnst.ROOT_DIR, args.model)
    eval_type = args.eval_type

    env = make_eval_env(
        env_id=env_name,
        dumpdir=helpers.resolve_video_dir(None, eval_type)
    )

    tf.reset_default_graph()

    print("Evaluating {}".format(model_path))

    is_monte = 'Monte' in env.unwrapped.spec._env_name

    model = Model(
        policy=policy_fn,
        ob_space=env.observation_space,
        ac_space=env.action_space
    )

    model.load(model_path)

    obs = env.reset()
    reset_needed = False

    total_rew = 0

    while True:
        act = model.eval_step(obs, eval_type)
        try:
            obs, _, done, info = env.step(act)
            total_rew += env.env._flat_reward
            env.render()

        except gym.error.ResetNeeded:
            reset_needed = True

        if reset_needed or (is_monte and info['ale.lives'] == 0):
            obs = env.reset()
            reset_needed = False

            print("Episode ended, total_rew = {}".format(total_rew))
            total_rew = 0


if __name__ == '__main__':
    args = atari_eval_arg_parser().parse_args()

    main(args)
