import csv
import cv2
import gym
import joblib
import numpy as np
import os
import tensorflow as tf
import time

from baselines.acktr.utils import find_trainable_variables
from baselines.common import tf_util
from baselines.common.cmd_util import make_eval_env

import helpers


class Model(object):

    def __init__(self, policy, ob_space, ac_space):

        sess = tf_util.make_session()
        step_model = policy(sess, ob_space, ac_space, 1, 1, reuse=False)
        params = find_trainable_variables("model")

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        def step(obs, eval_type):
            td_map = {step_model.X: [obs]}
            logits = sess.run(step_model.pi, td_map)[0]
            if eval_type == 'argmax':
                act = logits.argmax()
                if np.random.rand() < 0.01:
                    act = ac_space.sample()
                return act
            elif eval_type == 'prob':
                # probs = func(s[None, :, :, :])[0][0]
                x = logits
                e_x = np.exp(x - np.max(x))
                probs = e_x / e_x.sum(axis=0)
                act = np.random.choice(range(probs.shape[-1]), 1, p=probs)[0]
                return act
            else:
                raise ValueError("Unknown eval type {}".format(eval_type))

        def step_with_value(obs, eval_type):
            td_map = {step_model.X: [obs]}
            logits, value = sess.run([step_model.pi, step_model.vf], td_map)
            logits, value = logits[0], value[0]
            if eval_type == 'argmax':
                act = logits.argmax()
                if np.random.rand() < 0.01:
                    act = ac_space.sample()
                return act, value
            elif eval_type == 'prob':
                # probs = func(s[None, :, :, :])[0][0]
                x = logits
                e_x = np.exp(x - np.max(x))
                probs = e_x / e_x.sum(axis=0)
                act = np.random.choice(range(probs.shape[-1]), 1, p=probs)[0]
                return act, value
            else:
                raise ValueError("Unknown eval type {}".format(eval_type))

        self.step_model = step_model
        self.eval_step = step
        self.eval_step_value = step_with_value
        self.initial_state = step_model.initial_state
        self.load = load
        tf.global_variables_initializer().run(session=sess)


def easy_video(model, params, eval_type):
    tstart = time.time()
    print("Starting writing video...")

    vid_dir = helpers.resolve_video_dir(None, eval_type)

    if 'Doom' in params.env:
        import baselines.common.doom_cmd_util as stuff
        very_sparse = True
        env = stuff.make_doom_eval_env(dumpdir=vid_dir, very_sparse=very_sparse)
    else:
        env = make_eval_env(
            env_id=params.env,
            dumpdir=vid_dir
        )

    write_video(env, model, eval_type)
    model.save(os.path.join(vid_dir, 'current_model.npy'))

    duration = time.time() - tstart
    print("Finished writing video in {} to {}".format(duration, vid_dir))


def easy_eval_video(model, eval_type, storage_url, env_name, no_videos):
    tstart = time.time()
    print("Starting writing video...")

    vid_dir = helpers.resolve_video_dir(storage_url, eval_type)

    if 'Doom' in env_name:
        import baselines.common.doom_cmd_util as stuff
        very_sparse = 'VerySparse' in env_name
        env = stuff.make_doom_eval_env(dumpdir=vid_dir, very_sparse=very_sparse)
    else:
        env = make_eval_env(
            env_id=env_name,
            dumpdir=vid_dir
        )

    duration = time.time() - tstart
    print("Finished writing video in {} to {}".format(duration, vid_dir))

    return write_video(env, model, eval_type, vid_writes_to_go=no_videos)



def run_value_function_run(env, model, img_dir):
    """
    Write a series of images with annotated value function.
    env: deepmind_wrapped env
    model: loaded model
    img_dir: where the value images will be put
    """
    obs = env.reset()
    is_monte = 'Monte' in env.unwrapped.spec._env_name


    img_idx = 0
    all_done = False
    reset_needed = False

    print("Writing to {}".format(img_dir))
    total_reward = 0

    while not all_done:
        act, value = model.eval_step_value(obs, 'prob')
        img = obs[..., 3]

        text_img = cv2.putText(img=np.copy(img), text='{:.6f}'.format(value), org=(0, 10), fontFace=0, fontScale=0.33, color=255)
        cv2.imwrite(os.path.join(img_dir, 'img_{}.png'.format(img_idx)), text_img)
        try:
            obs, rew, done, info = env.step(act)
            total_reward += rew
        except gym.error.ResetNeeded:
            reset_needed = True

        if reset_needed or (is_monte and info['ale.lives'] == 0):
            all_done = True

        img_idx += 1

    print("Done! Total reward = {}".format(total_reward))


class DataGathering(object):
    """
    play_utils.play accepts a callback, here is excerpt from docs:
    callback: lambda or None
                     Callback if a callback is provided it will be executed after
                     every step. It takes the following input:
    obs_t: observation before performing action
    obs_tp1: observation after performing action
    action: action that was executed
    rew: reward that was received
    done: whether the environemnt is done or not
    info: debug info
    """
    def __init__(self, rood_data_dir):
        self.f = None
        self.root_data_dir = rood_data_dir
        self.reset_logging()

    def reset_logging(self):
        if self.f is not None:
            self.f.close()
        self.img_dir, csv_name = helpers.prepare_data_fnames(self.root_data_dir)
        self.f = open(csv_name, 'wt')
        self.logger = csv.DictWriter(self.f, fieldnames=('frame','reward','score','terminal', 'action', 'lifes'))
        self.logger.writeheader()

        self.img_id = 0
        self.score = 0

    def save_data(self, obs_t, obs_next, action, rew, done, info):
        img_path = os.path.join(self.img_dir, "{:07d}.png".format(self.img_id))
        cv2.imwrite(img_path, cv2.cvtColor(obs_next, cv2.COLOR_RGB2BGR))
        self.score += rew
        self.logger.writerow({
            'frame': self.img_id,
            'reward': rew,
            'score': self.score,
            'terminal': done,
            'action': action,
            'lifes': info['ale.lives']
        })

        # NOTE: If the framework is slow then this is the cause...
        self.f.flush()

        if done:
            self.reset_logging()

        self.img_id += 1


def run_data_gathering(env, model, root_data_dir, trajectories_to_go=30):
    """
    Write a series of unwrapped env images and log of what has happened

    env: deepmind_wrapped env
    model: loaded model
    img_dir: where the value images will be put
    """
    obs_t = env.reset()
    img_t = env.unwrapped._get_image()

    is_monte = 'Monte' in env.unwrapped.spec._env_name

    all_done = False
    reset_needed = False

    data_gatherer = DataGathering(root_data_dir)

    total_reward = 0

    while trajectories_to_go > 0:
        act, value = model.eval_step_value(obs_t, 'prob')

        try:
            obs_next, rew, done, info = env.step(act)
            img_next = env.unwrapped._get_image()
            data_gatherer.save_data(img_t, img_next, act, rew, done, info)
            obs_t, img_t = obs_next, img_next
            total_reward += rew
        except gym.error.ResetNeeded:
            reset_needed = True

        if reset_needed or (is_monte and info['ale.lives'] == 0):
            trajectories_to_go -= 1
            if trajectories_to_go > 0:
                obs_t = env.reset()
                img_t = env.unwrapped._get_image()
                reset_needed = False

            print("Trajectory written, reward = {}, {} to go".format(total_reward, trajectories_to_go))
            total_reward = 0


def write_video(env, model, eval_type, vid_writes_to_go=5):
    obs = env.reset()

    is_monte = (env.unwrapped.spec is not None) and ('Monte' in env.unwrapped.spec._env_name)

    total_rew = 0
    total_rewards = []
    reset_needed = False

    while vid_writes_to_go > 0:
        act = model.eval_step(obs, eval_type)
        try:
            obs, _, done, info = env.step(act)
            # rew = env.env._flat_reward
            # if np.abs(rew) > 1e-8:
            #     print("Got reward {}".format(rew))
            total_rew += env.env._flat_reward
        except gym.error.ResetNeeded:
            reset_needed = True

        if reset_needed or (is_monte and info['ale.lives'] == 0):
            vid_writes_to_go -= 1
            if vid_writes_to_go > 0:
                obs = env.reset()
                reset_needed = False

            print("Episode ended, total_rew = {}".format(total_rew))
            total_rewards.append(total_rew)
            total_rew = 0

    print("Finished writing video. Total rewards = {}".format(total_rewards))
    return total_rewards

