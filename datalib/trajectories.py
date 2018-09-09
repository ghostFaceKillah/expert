"""
Collection of utils for processing trajectory data.
"""

import numpy as np
import os
import pandas as pd


UGLY_HARDCODE = [
    '000017.csv', '000039.csv', '000020.csv', '000025.csv',
    '000024.csv', '000013.csv', '000032.csv', '000014.csv',
    '000035.csv', '000048.csv', '000045.csv', '000004.csv',
    '000019.csv', '000026.csv'
]

# 1, 3, 5, 14


def load_trajectories_by_score(trajectory_dir, max_score_cutoff, min_score_cutoff,
                               project_level_gamma, clip_rewards, frameskip, process_lost_lifes, use_n_trajectories=None):
    """
    trajectory_dir: dir with trajectory csvs
    max_score_cutoff: don't take transitions beyond this score, if None filtering doesn't apply
    min_score_cutoff: don't take trajectories with maximimum score below this level
    project_level_gamma: gamma set at the frameskipped env level - ususally 0.99
    clip_rewards: should we clip rewards to [-1, 1] interval
    frameskip: how much frameskip we use
    process_lost_lifes: Should we treat lost life as episode end in the A2C sense and set end reward to zero.
                       True for montezuma and for games with notion of 'life'.
    :return: big pandas.DataFrame with all transitions for training
    """
    gamma = np.power(project_level_gamma, 1./frameskip)

    acc = []

    all_trajectories = os.listdir(trajectory_dir)

    processed_trajectories = []

    if use_n_trajectories is None or use_n_trajectories == -1:
        trajectories = all_trajectories
    else:
        print("Will use less trajectories = {}".format(use_n_trajectories))
        trajectories = UGLY_HARDCODE[:use_n_trajectories]

    for short_fname in trajectories:
        if '.swp' not in short_fname:
            fname = os.path.join(trajectory_dir, short_fname)
            traj_name = short_fname.replace('.csv', '')
            df = pd.read_csv(fname)

            if len(df) == 0:
                continue

            df['trajectory'] = traj_name

            final_score = df.score.iloc[-1]

            if (min_score_cutoff is not None) and (final_score < min_score_cutoff):
                continue

            processed_trajectories.append((short_fname, final_score, len(df), float(final_score) / len(df)))

            if clip_rewards:
                df['reward'] = df.reward.clip(-1, 1)

            if max_score_cutoff is not None:
                df = df[df.score < max_score_cutoff]

            rewards_reverse_time = list(df.reward)[::-1]

            reward_acc = []
            running_reward = 0

            if process_lost_lifes:
                df['lost_life'] = df.lifes.diff() < 0
                lost_life_reverse_time = list(df.lost_life)[::-1]

                # Set gamma to be the same as in the A3C
                for reward, lost_life in zip(rewards_reverse_time, lost_life_reverse_time):
                    reward_acc.append(running_reward)
                    running_reward = reward + gamma * running_reward
                    if lost_life:
                        running_reward = 0

            else:
                # Set gamma to be the same as in the A3C
                for reward in rewards_reverse_time:
                    reward_acc.append(running_reward)
                    running_reward = reward + gamma * running_reward

            discounted_future_rewards = reward_acc[::-1]
            df.loc[:, 'future_rewards'] = discounted_future_rewards

            acc.append(df)

    acc_df = pd.concat(acc)
    print("Considering {} trajectories, in total {} transitions".format(len(processed_trajectories), len(acc_df)))
    return acc_df

