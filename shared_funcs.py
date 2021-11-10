import os

import numpy as np
import yaml


def read_config_yaml(config_filename="config.yaml", section=None):
    if not os.path.isfile(config_filename):
        raise FileNotFoundError(config_filename)
    with open(config_filename, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        if section is None:
            return config
        return config[section]


def write_config_yaml(data, config_filename):
    with open(config_filename, 'w') as ymlfile:
        yaml.dump(data, ymlfile, default_flow_style=False)


def angle_minuspitopi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_close(x, y, window):
    return np.abs(angle_minuspitopi(x - y)) <= window