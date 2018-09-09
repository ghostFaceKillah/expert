"""
Fast dataflows for training based on tensorpack.
"""

import cv2
import numpy as np
import os
import tqdm
import pandas as pd

# ugly import, but that's how the tensorpack rolls
from tensorpack import dataflow as tp_dataflow
from tensorpack.dataflow import *

import constants as cnst
import datalib.trajectories as trajectories
import datalib.data_downloader as data_downloader


class ExpertDataflow(RNGDataFlow):
    """
    Settings in config:
        exclude_zero_actions: should we exclude zero actions
        remap_actions: project actions to a smaller set of actions to
                       exclude meaningless actions?
        future_rewards: include future discounted rewards as data:
                        data tuples then are (img, action, future rewards)
    """
    def __init__(self, config):
        self.img_dir = config['img_dir']
        self.config = config

        if not os.path.isdir(self.img_dir):
            data_downloader.download_and_untar(
                config['gdrive_data_id'],
                config['data_dir']
            )

        self.img_df = trajectories.load_trajectories_by_score(
            trajectory_dir=self.config['traj_dir'],
            max_score_cutoff=self.config['max_score_cutoff'],
            min_score_cutoff=self.config['min_score_cutoff'],
            project_level_gamma=self.config['gamma'],
            clip_rewards=self.config['clip_rewards'],
            frameskip=self.config['frameskip'],
            process_lost_lifes=self.config_bool('process_lost_lifes', default=False),
            use_n_trajectories=self.config_bool('use_n_trajectories', default=None)
        )

        self.img_df.index = range(len(self.img_df))

        # Remap actions to a smaller action set
        if self.config_bool('remap_actions'):
            # This is Montezuma Revenge specific
            """
            action number -> action name 
            -> reduced action name -> reduced action number
            """
            self.img_df.loc[:, 'action'] = self.img_df.action.apply(
                lambda x: cnst.REL_ACT_NUM_MAP[
                    cnst.ACTION_TO_REL_ACT_MAP[
                        cnst.ACTION_NAMES[x]
                    ]
                ]
            )

        self.no_actions = len(set(self.img_df.action))

        # Return also discounted future rewards from the data stream
        # So the returned tuples are (img, action label, future reward)
        self.future_rewards = self.config_bool('future_rewards')

        # for debugging
        # self.img_df = self.img_df.iloc[-100:]

        # make sure we have sufficient frame history to go back
        # for example if frame_history is 4, we need to start at earliest
        # at frame 4, so we can take frames 0, 1, 2, 3
        # to predict actions 1, 2, 3, 4
        sel = self.img_df.frame >= self.config['frame_history'] * self.config['frameskip'] + 1

        self.sel = sel

        if self.config_bool('preload_images', default=False):
            self._preload_images()
        else:
            self._preloaded_images = {}

    def _preload_images(self):

        idxs = list(self.img_df.index)

        data = {}

        print("Preloading images")

        for idx in tqdm.tqdm(idxs):
            data[idx] = self._actually_load_image(idx)

        print("Done")

        self._preloaded_images = data

    def config_bool(self, name, default=False):
        """ is boolean present in config """
        if name in self.config:
            return self.config[name]
        else:
            return default

    def _setup_zero_action_sampling(self):
        if 'sample_zero_actions' in self.config:
            q = self.config['sample_zero_actions']
            allow_some_zero_actions = pd.Series(self.rng.uniform(0, 1, len(self.img_df)) < q, index=self.img_df.index)
            disallow_zero_actions = self.img_df.action != 0

            action_selector = allow_some_zero_actions | disallow_zero_actions
            self.sel = self.sel & action_selector

        if self.config_bool('exclude_zero_actions'):
            assert 'sample_zero_actions' not in self.config
            disallow_zero_actions = self.img_df.action != 0
            action_selector =  disallow_zero_actions
            self.sel = self.sel & action_selector

    def _load_image(self, the_idx):
        if the_idx in self._preloaded_images:
            return self._preloaded_images[the_idx]
        else:
            # print("Cache miss on the idx = {} ??? Weird".format(the_idx))
            thing = self._actually_load_image(the_idx)
            self._preloaded_images[the_idx] = thing
            return thing

    def _actually_load_image(self, the_idx):
        data_row = self.img_df.iloc[the_idx]

        trajectory = data_row.trajectory
        img_no = data_row.frame

        fname = os.path.join(
            self.img_dir,
            "{}".format(trajectory),
            "{:07d}.png".format(img_no)
        )

        # NOTE: Image reading can be pararellized via tensorpack dataflows
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        if self.config_bool('monte-specific-blackout'):
            img[0:23, ...] = 0
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.config_bool('pong-specific-blackout'):
            img[0:23, :] = [17, 72, 144]

        assert img is not None, fname

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)

        return img

    def load_stacked_imgs(self, current):
        # load images
        frames = []

        # make processed image stack
        for i in range(self.config['frame_history'])[::-1]:
            # -1, because we take _previous_ observation to predict action
            img_one = self._load_image(current - i * self.config['frameskip'] - 1)
            img_two = self._load_image(current - i * self.config['frameskip'] - 2)

            buffer = np.zeros((2,)+img_one.shape, dtype=np.uint8)
            buffer[0] = img_one
            buffer[1] = img_two
            img = buffer.max(axis=0)

            frames.append(img[..., np.newaxis])

        imgs = np.concatenate(frames, axis=2)

        return imgs

    def get_data(self):
        # We run it only now, because we need self.rng to be reset.
        self._setup_zero_action_sampling()

        idxs = list(self.img_df[self.sel].index)

        self.rng.shuffle(idxs)
        idx = 0

        while True:

            if idx == len(idxs):
                self.rng.shuffle(idxs)
                idx = 0

            # choose row
            current = idxs[idx]

            imgs = self.load_stacked_imgs(current)
            data_row = self.img_df.iloc[current]
            future_rewards = data_row.future_rewards
            action_label = data_row.action

            idx += 1

            # return the wanted variant of data
            if self.future_rewards:
                yield [imgs, action_label, future_rewards]
            elif self.action_label:
                yield [imgs, action_label]


def get_dataflows(config):
    """
    construct and initialize dataflows based on config.
    """

    df = ExpertDataflow(config)
    df = tp_dataflow.PrefetchDataZMQ(df, nr_proc=16)
    df = tp_dataflow.BatchData(df, config['batch_size'], remainder=False)

    # initialize random number generator in child processes to unique values
    df.reset_state()

    return df