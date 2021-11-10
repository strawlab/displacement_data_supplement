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


def angle_in_range(angle, angle_from, angle_to):
    if abs(angle_to - angle_from) >= 2 * np.pi:
        return True
    afrom = angle_minuspitopi(angle_from)
    ato = angle_minuspitopi(angle_to)
    a = angle_minuspitopi(angle)
    reversing = ((ato - afrom) * (angle_to - angle_from) < 0)
    #     print("reversing", reversing)
    if not reversing:
        #         print(afrom, ato)
        #         print(a-ato, a-afrom)
        return (a - ato) * (a - afrom) < 0
    if reversing:
        #         print(a-ato, a-afrom)
        a = angle_minuspitopi(a + np.pi)
        #         print(a-ato, a-afrom)
        return (a - ato) * (a - afrom) < 0
