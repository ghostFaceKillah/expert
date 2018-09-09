#!/usr/bin/env python3

import numpy as np

from baselines import logger
from baselines.acktr.acktr_disc import learn
from baselines.acktr.policies import CnnPolicy
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

import constants as cnst


def train(params):
    policy_fn = CnnPolicy

    dataflow_config = {
        'future_rewards': True,             # Should return future discounted rewards?
        'exclude_zero_actions': False,      # Should exclude zero actions
        'remap_actions': False,             # Should remap to smaller action set?
        'clip_rewards': True,               # Clip rewards to [-1, 1]
        'monte-specific-blackout': True,    # Cover up score and lives indicators
        'pong-specific-blackout': False,    # Cover up scores in pong
        'gamma': params.gamma,              # reward discount factor
        'frame_history': 4,                 # What is minimum number of expert frames since beginning of episode?
        'frameskip': 4,                     # frameskip
        'preload_images': True,             # Preload images from hard drive or keep reloading ?
        'gdrive_data_id': cnst.MONTE_DATA_GDRIVE_ID,
        'data_dir': cnst.DATA_DIR,
        'img_dir': cnst.MIKE_IMG_DIR,
        'traj_dir': cnst.MIKE_TRAJECTORIES_DIR,
        'stat_dir': cnst.MIKE_STATES_DIR,
        'batch_size': params.expert_nbatch,
        'max_score_cutoff': params.exp_max_score,  # What is maximum expert score we can show? Used to cut expert data
        'min_score_cutoff': 20000,                 # What is minimum score to count trajectory as expert
        'process_lost_lifes': True,                # Should loss of life zero future discounted reward?
        'use_n_trajectories': params.use_n_trajectories if 'use_n_trajectories' in params else None
    }

    the_seed = np.random.randint(10000)
    print(80 * "SEED")
    print("Today's lucky seed is {}".format(the_seed))
    print(80 * "SEED")

    env = VecFrameStack(
        make_atari_env(
            env_id=params.env,
            num_env=params.num_env,
            seed=the_seed,
            limit_len=params.limit_len,
            limit_penalty=params.limit_penalty,
            death_penalty=params.death_penalty,
            step_penalty=params.step_penalty,
            random_state_reset=params.random_state_reset,
            dataflow_config=dataflow_config
        ),
        params.frame_stack
    )

    learn(
        policy=policy_fn,
        env=env,
        seed=the_seed,
        params=params,
        dataflow_config=dataflow_config,
        expert_nbatch=params.expert_nbatch,
        exp_adv_est=params.exp_adv_est,
        load_model=params.load_model,
        gamma=params.gamma,
        nprocs=params.num_env,
        nsteps=params.nsteps,
        ent_coef=params.ent_coef,
        expert_coeff=params.exp_coeff,
        lr=params.lr,
        lrschedule=params.lrschedule,
    )

    env.close()


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    params = atari_arg_parser().parse_args()
    logger.configure(dir=cnst.openai_logdir())
    train(params)


if __name__ == '__main__':
    main()
