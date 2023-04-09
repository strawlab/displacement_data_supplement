import os

import numpy as np
import pandas as pd
import sys

from pheromone_model import ChannelWithPheromones, MyFlyPheromones
from shared_funcs import read_config_yaml, write_config_yaml

if __name__ == '__main__':
    data_filename = sys.argv[1]
    big_config = read_config_yaml(config_filename="config_rew3.yaml")

    data_folder = big_config["data_folder"]
    n_simulations = big_config["iterations"]

    model_config = big_config["model_settings"][data_filename]

    save_config_to = f"{data_filename[:-4]}.yaml"
    my_config_filename = os.path.join(data_folder, save_config_to)

    dfs = []
    last_foods = [0, 1, 2]
    disable_food_time = 300*2

    channel_hl = big_config['channel']['half_length']
    reward_refractory_period = big_config['channel']['refractory_period']

    for last_food in last_foods:
        print(f"Food index: {last_food} ... ")
        for i in range(n_simulations):
            channel = ChannelWithPheromones(enable_food_time=5, disable_food_time=None,
                                            food_coords=[0, 5 * np.pi / 26, -5 * np.pi / 26],
                                            refractory_period=reward_refractory_period, channel_hl=channel_hl)
            # channel.set_config(model_config)
            # channel = ChannelWithPheromones(enable_food_time=5, disable_food_time=300*2)
            channel.schedule[5] = {0: True, 1: True, 2: True}

            # disable all but 1 food sources
            for foodi in [0, 1, 2]:
                if foodi != last_food:
                    channel.schedule[disable_food_time][foodi] = False

            channel.time_off = disable_food_time
            channel.last_food_index = last_food

            # print(channel.schedule)

            fly = MyFlyPheromones(channel, model_config=model_config)
            fly.start_walking(Tlim=600*2)

            flydf = fly.get_df()
            flydf["flyid"] = i*10 + last_food
            flydf["last_food_coord"] = channel.food[last_food]
            flydf["last_food_index"] = last_food
            dfs.append(flydf)

    df = pd.concat(dfs, ignore_index=True)
    csv_fname = os.path.join(data_folder, data_filename)
    print(f"saving data {df.shape} to {csv_fname}")
    df.to_csv(csv_fname, index=False)
    write_config_yaml(model_config, my_config_filename)

