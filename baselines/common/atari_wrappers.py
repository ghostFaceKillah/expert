import numpy as np
from collections import deque
import gym
import os
import gym.spaces as spaces
import cv2

import datalib.trajectories as trajectories

cv2.ocl.setUseOpenCL(False)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class SavedClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._flat_reward = 0

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        self._flat_reward = reward
        return np.sign(reward)

class SavedPositiveClippedRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._flat_reward = 0

    def reward(self, reward):
        """Bin reward to {+1, 0} by its sign."""
        self._flat_reward = reward
        return max(np.sign(reward), 0)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, is_monte, is_pong):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.is_monte = is_monte
        self.is_pong = is_pong
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        if self.is_monte:
            frame[0:23, ...] = 0
        if self.is_pong:
            frame[0:23, :] = [144, 72, 17]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class PenalizeDying(gym.Wrapper):
    """
    {'ale.lives': 6}
    """
    def __init__(self, env, penalty):
        gym.Wrapper.__init__(self, env)
        self.lives = 6
        self.penalty = penalty

    def reset(self):
        ob = self.env.reset()
        self.lives = 6
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        new_lives = info['ale.lives']

        if new_lives < self.lives:
            self.lives = new_lives
            reward -= self.penalty
            # done = True

        return ob, reward, done, info


class StepPenalty(gym.Wrapper):
    def __init__(self, env, step_penalty):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.step_penalty = step_penalty

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        reward = reward - self.step_penalty
        return ob, reward, done, info



class LimitLength(gym.Wrapper):
    def __init__(self, env, k, timeout_penalty):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.timeout_penalty = timeout_penalty

    def reset(self):
        # This assumes that reset() will really reset the env.
        # If the underlying env tries to be smart about reset
        # (e.g. end-of-life), the assumption doesn't hold.
        ob = self.env.reset()
        self.cnt = 0
        return ob

    def step(self, action):
        ob, r, done, info = self.env.step(action)
        self.cnt += 1
        if self.cnt == self.k:
            done = True
            r -= self.timeout_penalty
        return ob, r, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class RandomStartingWrapper(gym.Wrapper):
    def __init__(self, env, config):
        super(RandomStartingWrapper, self).__init__(env)

        self.config = config

        self.df = trajectories.load_trajectories_by_score(
            trajectory_dir=config['traj_dir'],
            max_score_cutoff=config['max_score_cutoff'],
            min_score_cutoff=config['min_score_cutoff'],
            project_level_gamma=config['gamma'],
            clip_rewards=config['clip_rewards'],
            frameskip=config['frameskip'],
            process_lost_lifes=config['process_lost_lifes'],
        )

        self.random_state = None

    def seed(self, seed=None):
        self.env.seed(seed)

        if seed is None:
            raise ValueError("Seed cannot be None in case of random starting env wrapper")

        self.random_state = np.random.RandomState(seed)

    def reset(self, **kwargs):
        super(RandomStartingWrapper, self).reset(**kwargs)
        wrapped_env = self.env.env

        if self.random_state is None:
            raise ValueError("Uninitialized random state")

        idx = self.random_state.randint(1, len(self.df))

        # We have to kick out the first frame, because we don't have observation before it
        while self.df.iloc[idx].frame == 0:
            idx = self.random_state.randint(1, len(self.df))

        # print("Will restore state no = {}".format(idx))

        traj = self.df.iloc[idx].trajectory
        state_idx = self.df.iloc[idx].frame

        state_fname = os.path.join(self.config['stat_dir'], "{}/{:07d}.npy".format(traj, state_idx))
        state = np.load(state_fname)

        img_fname = os.path.join(self.config['img_dir'], "{}/{:07d}.png".format(traj, state_idx - 1))
        img = cv2.imread(img_fname, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        wrapped_env.restore_full_state(state)

        # wrapped_env._get_obs() returns observation before state change, so we have to fix it ourselves
        # https://github.com/openai/gym/issues/715

        return img


class DoomMyWayHomeActionWrapper(gym.ActionWrapper):
    """
    Doom my way home env (see doom.env.doom_my_way_home). has action space:
        actions = [0] * 43
        actions[13] = 0      # MOVE_FORWARD
        actions[14] = 1      # TURN_RIGHT
        actions[15] = 0      # TURN_LEFT

    We need to change that to match the scheme I have implemnted while gathering data
    (and to much the stoachastic policy reinforecement learning formulation).

    We want to map e.g.:
        7 -> [1, 1, 1]
        5 -> [1, 0, 1]

    (but ofc the relevant array starts from place 13)
    """
    def __init__(self, env):
        super(DoomMyWayHomeActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(8)

    def action(self, action):
        a = action
        move_fwd = a % 2
        a = a // 2
        turn_right = a % 2
        a = a // 2
        turn_left = a % 2
        a = a // 2
        assert a == 0

        out = [0] * 43
        out[0] = move_fwd
        out[1] = turn_right
        out[2] = turn_left

        return out


def make_state_restoring_atari(env_id, config):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = RandomStartingWrapper(env, config)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False, is_monte=False, is_pong=False, save_original_reward=False, only_positive_rewards=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, is_monte, is_pong)
    if scale:
        env = ScaledFloatFrame(env)
    if only_positive_rewards:
        env = SavedPositiveClippedRewardEnv(env)
    elif clip_rewards:
        if save_original_reward:
            env = SavedClipRewardEnv(env)
        else:
            env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


def wrap_doom_deepmind_like(env, clip_rewards=True, frame_stack=False, scale=False, save_original_reward=False):
    env = WarpFrame(env, is_monte=False, is_pong=False)
    env = DoomMyWayHomeActionWrapper(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        if save_original_reward:
            env = SavedClipRewardEnv(env)
        else:
            env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


