from collections import defaultdict

import numpy as np

from shared_funcs import angle_close

CHANNEL_HL = 26  # default value of channel half length (in BL == steps)


class ChannelEnvironment:
    def __init__(self, food_coords=[0], refractory_period=16, channel_hl=CHANNEL_HL, **kwargs):
        """
        :param food_coords:
        :param enable_food_time:
        :param disable_food_time:
        :param refractory_period: time of food deactivation after the fly encountered it. Defalt: 16 steps (8 sec)
        """
        self.channel_hl = channel_hl
        self.refractory_period = refractory_period
        print("Channel half length: ", self.channel_hl)

        self.food_w = np.pi / self.channel_hl / 2.  # half-width of the food., 2xfood_w = BL, same as step size.
        self.food = np.array(food_coords)  # list of food locations
        self.food_enabled = np.array([False] * len(food_coords))  # all food locations disabled

        self.schedule = defaultdict(dict)
        self.set_schedule(**kwargs)

        self.fly_on_food = None
        self.last_t = 0

        self.food_log = defaultdict(list)
        #{"t": [], 0: []}

        self.last_food_index = None
        self.time_off = None

    def set_schedule(self, enable_food_time=5, disable_food_time=100, **kwargs):
        food_on_dict = {}
        food_off_dict = {}
        if disable_food_time is not None:  # do not set schedule if disable food is None
            for foodi in range(len(self.food)):
                food_on_dict[foodi] = True
                food_off_dict[foodi] = False

            self.schedule[enable_food_time] = food_on_dict
            self.schedule[disable_food_time] = food_off_dict

    def env_state_update(self, t):  # when the food is active
        self.last_t = t
        if t in self.schedule.keys():
            actions = self.schedule[t]
            for foodid, state in actions.items():
                self.set_enabled_food(foodid, state)
            del self.schedule[t]

        self.food_log["t"].append(self.last_t)
        for ifood in range(len(self.food)):
            self.food_log[ifood].append(self.food_enabled[ifood])

    def update(self, fly_coord, t):
        self.env_state_update(t)
        food_locs, food_indices = self.get_enabled_food_locations()
        for food_index, food_loc in zip(food_indices, food_locs):
            # print("update:", t, food_loc, fly_coord,
            #       "fly-food:", fly_coord - food_loc,
            #       "-pi to pi:", angle_minuspitopi(fly_coord - food_loc))
            if angle_close(fly_coord, food_loc, window=self.food_w):
                self.fly_on_food = food_index
                return self.fly_on_food
        self.fly_on_food = None
        return self.fly_on_food

    def get_enabled_food_locations(self):
        return self.food[self.food_enabled], self.food_enabled.nonzero()[0]  # locations, indices

    def set_enabled_food(self, food_i, enabled_value):
        self.food_enabled[food_i] = enabled_value
        # print(f"t={self.last_t}: Food #{food_i} [{self.food[food_i]}] := {enabled_value}")

    def enable_food(self, food_i=0):
        self.set_enabled_food(food_i, True)

    def disable_food(self, food_i=0, refractory=False):
        self.set_enabled_food(food_i, False)

        # if there is a food source which is active 1 time more than the others, don't turn it on again after time_off
        if self.last_food_index == food_i and self.last_t > self.time_off:
            print(f"t={self.last_t}, food {food_i} last turn off!")
            refractory = False

        if refractory:
            # don't turn the food on if the schedule says it should be turned off during refractory period.
            for tbetween in range(self.last_t, self.last_t + self.refractory_period + 1):
                if tbetween in self.schedule and not self.schedule[tbetween].get(food_i, True):
                    return
            # otherwise schedule food on after refractory period
            self.schedule[self.last_t + self.refractory_period][food_i] = True

    def print_current_state(self):
        print("food:\n", np.vstack([self.food, self.food_enabled]), "\n")
