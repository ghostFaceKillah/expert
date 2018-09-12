"""
Project-wide constants
"""

import datetime
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_IMG_SIZE = (210, 160, 3)
ACTION_SPACE_SIZE = 18

MONTE_DATA_GDRIVE_ID = '1Q6rgqLEr9JIqFDNkYnNn93vBkyEUDPF6'

def openai_logdir():
    return os.path.join(
        ROOT_DIR,
        'openai-logs',
        datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f")
    )


def joblib_cache_dir():
    return os.path.join(
        ROOT_DIR,
        'joblib-cache',
    )


ACTION_NAMES = [
    'NOOP',
    'FIRE',
    'UP',
    'RIGHT',
    'LEFT',
    'DOWN',
    'UPRIGHT',
    'UPLEFT',
    'DOWNRIGHT',
    'DOWNLEFT',
    'UPFIRE',
    'RIGHTFIRE',
    'LEFTFIRE',
    'DOWNFIRE',
    'UPRIGHTFIRE',
    'UPLEFTFIRE',
    'DOWNRIGHTFIRE',
    'DOWNLEFTFIRE'
]

ACTION_NAME_TO_NUM = {name: idx for idx, name in enumerate(ACTION_NAMES)}

DATA_DIR = os.path.join(ROOT_DIR, 'in_data')

MIKE_TRAJECTORIES_DIR = os.path.join(ROOT_DIR, 'in_data', 'our-data', 'trajectories')

PITFALL_RIGHT_TRAJECTORIES_DIR = os.path.join(ROOT_DIR, 'in_data', 'pitfall-right', 'trajectories')
PITFALL_RIGHT_IMG_DIR = os.path.join(ROOT_DIR, 'in_data', 'pitfall-right', 'screens')
PITFALL_RIGHT_STATES_DIR = os.path.join(ROOT_DIR, 'in_data', 'pitfall-right', 'states')

EVAL_DATA_PATH = os.path.join(ROOT_DIR, 'eval_data')

MIKE_IMG_DIR = os.path.join(ROOT_DIR, 'in_data', 'our-data', 'screens')
MIKE_STATES_DIR = os.path.join(ROOT_DIR, 'in_data', 'our-data', 'states')

DOOM_DIR = os.path.join(ROOT_DIR, 'in_data', 'doom')
DOOM_TRAJ_DIR = os.path.join(DOOM_DIR, 'trajectories')
DOOM_IMG_DIR = os.path.join(DOOM_DIR, 'screens')

VERY_SPARSE_DOOM_DIR = os.path.join(ROOT_DIR, 'in_data', 'doom-very-sparse')
VERY_SPARSE_DOOM_TRAJ_DIR = os.path.join(VERY_SPARSE_DOOM_DIR, 'trajectories')
VERY_SPARSE_DOOM_IMG_DIR = os.path.join (VERY_SPARSE_DOOM_DIR, 'screens')


# The rest of the actions are essentially replications of the below actions...
RELEVANT_ACTIONS = [
    'FIRE',
    'UP',
    'RIGHT',
    'LEFT',
    'DOWN',
    'RIGHTFIRE',
    'LEFTFIRE',
    'NOOP',
]

REL_ACT_TO_ACTION_NUM_MAP = {
    'NOOP': 0,
    'FIRE': 1,
    'UP': 2,
    'RIGHT': 3,
    'LEFT': 4,
    'DOWN': 5,
    'RIGHTFIRE': 11,
    'LEFTFIRE': 12
}

REL_ACT_NUM_MAP = {
    'FIRE': 0,
    'UP': 1,
    'RIGHT': 2,
    'LEFT': 3,
    'DOWN': 4,
    'RIGHTFIRE': 5,
    'LEFTFIRE': 6,
    'NOOP': 7,
}

REL_ACT_REVERSE_MAP = dict([(t[1], t[0]) for t in REL_ACT_NUM_MAP.items()])

ACTION_TO_REL_ACT_MAP = {
    'NOOP': 'NOOP',

    'LEFT': 'LEFT',
    'RIGHT': 'RIGHT',

    'UP': 'UP',
    'UPLEFT': 'UP',
    'UPRIGHT': 'UP',

    'DOWN': 'DOWN',
    'DOWNLEFT': 'DOWN',
    'DOWNRIGHT': 'DOWN',

    'LEFTFIRE': 'LEFTFIRE',
    'UPLEFTFIRE': 'LEFTFIRE',
    'DOWNLEFTFIRE': 'LEFTFIRE',

    'RIGHTFIRE': 'RIGHTFIRE',
    'UPRIGHTFIRE': 'RIGHTFIRE',
    'DOWNRIGHTFIRE': 'RIGHTFIRE',

    'FIRE': 'FIRE',
    'UPFIRE': 'FIRE',
    'DOWNFIRE': 'FIRE',

}
