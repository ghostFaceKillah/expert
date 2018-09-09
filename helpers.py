import cv2
import datetime
import os

import constants as cnst


def dump_obs(obs):
    for i in range(4):
        cv2.imwrite('hehe_{}.png'.format(i), obs[..., i])
    print("Written to drive!")


def resolve_video_dir(neptune_storage_url, eval_type):
    if neptune_storage_url is not None:
        root_dir = neptune_storage_url
    else:
        root_dir = cnst.ROOT_DIR

    dir_stub = 'vid' if eval_type == 'prob' else 'vid-argmax'

    return os.path.join(root_dir, dir_stub, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))


def resolve_out_data_dir(neptune_storage_url):
    if neptune_storage_url is not None:
        root_dir = neptune_storage_url
    else:
        root_dir = cnst.ROOT_DIR

    return os.path.join(root_dir, 'out_data', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))


def best_model_save_path(neptune_storage_url):
    if neptune_storage_url is not None:
        root_dir = neptune_storage_url
    else:
        root_dir = cnst.ROOT_DIR

    models_path = os.path.join(root_dir, 'models')
    mkdir_p(models_path)

    return os.path.join(models_path, 'best_model.npy')


def model_save_path(neptune_storage_url):
    if neptune_storage_url is not None:
        root_dir = neptune_storage_url
    else:
        root_dir = cnst.ROOT_DIR

    models_path = os.path.join(root_dir, 'models')
    mkdir_p(models_path)

    models_no = [
        int(f.split('_')[1]) for f in os.listdir(models_path)
        if 'model' in f and 'best' not in f
    ]
    if len(models_no) == 0:
        max_model_no = 0
    else:
        max_model_no = max(models_no)

    model_fpath = os.path.join(
        models_path,
        "model_{:03d}_{}.npy".format(max_model_no + 1, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
    )

    return model_fpath


def mkdir_p(dir):
    """ Check if directory exists and if not, make it."""
    if not os.path.exists(dir):
        os.makedirs(dir)


def _get_next_traj_id(root_data_dir='data'):
    """ Resolve what is the next trajectory number """
    if not os.path.exists(os.path.join(root_data_dir, 'screens')):
        return 0
    return 1 + max([
        int(x) for x in os.listdir(os.path.join(root_data_dir, 'screens'))
    ])


def prepare_data_fnames(root_data_dir='data'):
    """ Easy prepare paths to which we will write generated training data  """
    traj_no = _get_next_traj_id(root_data_dir)
    screen_dir = os.path.join(root_data_dir, 'screens', "{:06d}".format(traj_no))

    if not os.path.exists(screen_dir):
        os.makedirs(screen_dir)

    csv_dir = os.path.join(root_data_dir, 'trajectories')

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    csv_name = os.path.join(csv_dir, "{:06d}.csv".format(traj_no))

    return screen_dir, csv_name

