import os

import numpy as np
import pandas as pd
import sys

import yaml

from pheromone_model import ChannelWithPheromones, MyFlyPheromones


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


if __name__ == '__main__':
    data_filename = sys.argv[1]
    big_config = read_config_yaml()
    config = big_config[data_filename]
    my_config_filename = f"data/{data_filename[:-4]}.yaml"
    dfs = []
    n_simulations = big_config["iterations"]
    last_foods = [0, 1, 2]
    disable_food_time = 300*2
    for last_food in last_foods:
        print(f"Food index: {last_food} ... ")
        for i in range(n_simulations):
            channel = ChannelWithPheromones(enable_food_time=5, disable_food_time=None,
                                            food_coords=[0, 5 * np.pi / 26, -5 * np.pi / 26])
            channel.set_config(config)
            # channel = ChannelWithPheromones(enable_food_time=5, disable_food_time=300*2)
            channel.schedule[5] = {0: True, 1: True, 2: True}

            # disable all but 1 food sources
            for foodi in [0, 1, 2]:
                if foodi != last_food:
                    channel.schedule[disable_food_time][foodi] = False

            channel.time_off = disable_food_time
            channel.last_food_index = last_food

            # print(channel.schedule)

            fly = MyFlyPheromones(channel)
            fly.start_walking(Tlim=600*2)

            flydf = fly.get_df()
            flydf["flyid"] = i*10 + last_food
            flydf["last_food_coord"] = channel.food[last_food]
            flydf["last_food_index"] = last_food
            dfs.append(flydf)

    df = pd.concat(dfs, ignore_index=True)
    csv_fname = os.path.join("data", data_filename)
    print(f"saving data {df.shape} to {csv_fname}")
    df.to_csv(csv_fname, index=False)
    write_config_yaml(config, my_config_filename)
