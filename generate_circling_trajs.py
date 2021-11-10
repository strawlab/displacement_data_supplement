import os
import sys
import pandas as pd
import numpy as np

from shared_funcs import read_config_yaml, write_config_yaml
from pheromone_model import ChannelWithPheromones, MyFlyPheromones

if __name__ == '__main__':
    data_filename = sys.argv[1]

    big_config = read_config_yaml(config_filename="config_circling.yaml")

    data_folder = big_config["data_folder"]
    n_simulations = big_config["iterations"]

    model_config = big_config["model_settings"][data_filename]
    print(model_config)
    print("-----------------------------------------")
    save_config_to = f"{data_filename[:-4]}.yaml"
    my_config_filename = os.path.join(data_folder, save_config_to)

    dfs = []

    each_fly_iterations = 6
    AP_duration = 5*60*2  # 5 minutes, one time step is 0.5 sec
    postAP_duration = 5 * 60 * 2  # 5 minutes, one time step is 0.5 sec
    start_t = 5

    channel_hl = big_config['channel']['half_length']
    reward_refractory_period = big_config['channel']['refractory_period']
    for i in range(n_simulations):
        channel = ChannelWithPheromones(enable_food_time=None, disable_food_time=None,
                                        food_coords=[0, np.pi/2],
                                        refractory_period=reward_refractory_period, channel_hl=channel_hl)

        #channel.schedule[0] = {0: True, 1: False}
        # 6 iterations for one fly
        for iter_num in range(each_fly_iterations):
            right_on = (iter_num % 2 == 0)
            channel.schedule[(AP_duration+postAP_duration)*iter_num + start_t] = {0: right_on, 1: not right_on}
            channel.schedule[(AP_duration + postAP_duration) * iter_num + AP_duration + start_t] = {0: 0, 1: 0}

        print(channel.schedule)

        fly = MyFlyPheromones(channel, model_config=model_config)
        fly.start_walking(Tlim=(AP_duration + postAP_duration)*each_fly_iterations+start_t)

        flydf = fly.get_df()
        flydf["flyid"] = i
        dfs.append(flydf)

    df = pd.concat(dfs, ignore_index=True)
    csv_fname = os.path.join(data_folder, data_filename)
    print(f"saving data {df.shape} to {csv_fname}")
    df.to_csv(csv_fname, index=False)
    write_config_yaml(model_config, my_config_filename)
