import numpy as np
import matplotlib.pyplot as plt

from fr_model import ChannelEnvironment, FLY_EATING_TIME, angle_close, angle_minuspitopi
import pandas as pd

# after food
RL_mean = 4.125
RL_std = 2.625

# baseline
#BLRL_mean = 10 * RL_mean
#BLRL_std = 2 * RL_std
BLRL_mean = 50
BLRL_std = 5



class ChannelWithPheromones(ChannelEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pheromone_w = self.food_w
        self.pheromone_dict = {}
        self.odor=0

    def add_pheromone(self, coord, lifetime, strength_init=1):
        self.pheromone_dict[coord] = {'lifetime': lifetime,
                                      'strength_init': strength_init,
                                      'strength': strength_init,
                                      'time_added': self.last_t}

    def env_state_update(self, t):
        super().env_state_update(t)  # updates food states
        for pcoord, pstate in self.pheromone_dict.items():
            s0 = pstate['strength_init']
            pstate['strength'] = s0 - s0/pstate['lifetime']*(t - pstate["time_added"])  # 0 to s0
            if pstate['strength'] <= 0:
                pstate['strength'] = 0

    def is_fly_on_food(self, fly_coord):
        food_locs, food_indices = self.get_enabled_food_locations()
        for food_index, food_loc in zip(food_indices, food_locs):
            print("update:", self.last_t, food_loc, fly_coord,
                  "fly-food:", np.abs(fly_coord - food_loc),
                  "-pitopi:", angle_minuspitopi(fly_coord - food_loc))
                  #"mod2pi:", np.abs(fly_coord - food_loc) % (2 * np.pi))
            if angle_close(fly_coord, food_loc, window=self.food_w):
                ret = food_index
                return ret
        return None

    def smell_pheromones(self, fly_coord):
        smell = 0
        for pcoord, pstate in self.pheromone_dict.items():
            if pstate['strength'] <= 0:
                continue
            if angle_close(fly_coord, pcoord, window=self.pheromone_w):
            #if np.abs(fly_coord - pcoord) % (2 * np.pi) <= self.pheromone_w:
                smell = pstate['strength']
        return smell

    def update(self, fly_coord, t):
        self.env_state_update(t)
        self.fly_on_food = self.is_fly_on_food(fly_coord)
        self.odor = self.smell_pheromones(fly_coord)
        print(t, "fly on food:", self.fly_on_food, ", odor:", self.odor)


class MyFlyPheromones:
    def __init__(self, myenv: ChannelWithPheromones = None):
        self.eating_time = FLY_EATING_TIME
        if myenv is None:
            self.environment = ChannelWithPheromones()
        else:
            self.environment = myenv
        self.phi_step = np.pi / self.environment.channel_hl  # 1 body length step in radians depending on channel length

        self.t = 0

        self.run_length = None

        # mind
        self.state = 'walking'  # or eating
        self.mode = 'GS'  # or LS
        self.direction = 1  # or -1
        self.last_state = None

        self.current_run = 0  # distance integrator (step)

        # position in the environment
        self.coord_x = 0
        self.coord_y = 0
        self.coord_phi = - 8 * self.phi_step

        # log to save history, for plotting
        self.t_log = [0]
        self.phi_log = [self.coord_phi]
        self.eat_log = [False]
        self.smell_log = [False]
        self.direction_log = [self.direction]

    def log(self, eating=False, smelling=False):
        self.t_log.append(self.t)
        self.phi_log.append(self.coord_phi)
        self.eat_log.append(eating)
        self.smell_log.append(smelling)
        self.direction_log.append(self.direction)

    def make_step(self, direction):
        self.t += 1
        d_angle = self.phi_step * direction
        self.coord_phi += d_angle
        self.coord_x = np.cos(self.coord_phi)
        self.coord_y = np.sin(self.coord_phi)
        dx = np.cos(d_angle)
        dy = np.sin(d_angle)
        self.current_run += np.sqrt(dx ** 2 + dy ** 2)  # +=1

        smelling=False

        # index of food the fly is on at the moment, None if not on an active food
        self.environment.update(self.coord_phi, self.t)
        if self.environment.fly_on_food is not None:
            self.on_food(self.environment.fly_on_food)
        elif self.environment.odor != 0:
            self.feel_pheromones(self.environment.odor)
            smelling = True

        self.log(smelling=smelling)  # moved from before updating environment

    def plot_angle_history(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.t_log, self.phi_log, '.-')

    def plot_trajectory(self, ax=None):
        if ax is None:
            ax = plt.gca()

        angles = np.array(self.phi_log)
        xx = np.cos(angles)
        yy = np.sin(angles)
        ax.plot(xx, yy)
        ax.axis('equal')

    def on_food(self, food_index):
        print(self.t, " - fly on food!")
        self.mode = 'LS'  # enter the local search mode

        # remember the eating state
        self.last_state = 'eating'

        # disable food source for refractory period
        self.environment.disable_food(food_index, refractory=True)

        # stay on food location for a while
        self.eat()
        self.release_pheromone()
        self.choose_run_length()

    def eat(self):
        # do nothing for self.eating time (only update the environment based on current time)
        for t in range(self.eating_time):
            self.t += 1
            # update log at every step
            self.log(eating=True)
            # update environment
            # print('fly eating', self.t)
            self.environment.update(self.coord_phi, self.t)

    def choose_run_length(self, odor=0):
        # if just was on food
        if self.last_state == 'eating':
            #self.run_length = np.random.normal(RL_mean, RL_std) + self.current_run
            self.run_length = np.random.normal(RL_mean, RL_std)  # not based on current run length, choose new.

        elif self.last_state == 'reversal':
            #self.run_length = np.abs(self.current_run + np.random.normal(dRL_mean, dRL_std))
            self.run_length = np.random.normal(BLRL_mean, BLRL_std)  # choose big run length
        elif self.last_state == 'smelling':
            # change mean run length based on odor value: if odor=1, same as food, closer to 0 - 3 times longer walk.
            rl_mean = RL_mean * (1 + 2*(1-odor))
            rl_std = RL_std*(2-odor)
            self.run_length = np.random.normal(rl_mean, rl_std)

        print(f"Prev: {self.last_state}, last run was: {self.current_run}, RL:{self.run_length}")
        self.current_run = 0  # start walking from here, forget the past.

    def start_walking(self, Tlim=500):
        while self.mode == "GS":
            self.make_step(self.direction)
        if self.mode == 'LS':
            while self.t < Tlim:
                # print(f"{self.t} | run: {self.current_run}")
                self.make_step(self.direction)
                if self.current_run >= self.run_length:
                    self.reversal()

    def reversal(self):
        self.direction *= -1
        self.last_state = 'reversal' # moved from the end to before choosing new run len
        self.choose_run_length()
#        self.zero_integrator() moved it to choose_run_length

    def get_df(self):
        df = pd.DataFrame(dict(t=self.t_log, angle=self.phi_log,
                                 eating=self.eat_log, direction=self.direction_log,
                               smelling=self.smell_log))
        run_num = df.direction.diff().abs()/2
        df["run_num"] = run_num.cumsum()
        df.loc[0, "run_num"] = 0
        df.run_num = df.run_num - df[df.eating].iloc[-1].run_num - 1
        return df

    def feel_pheromones(self, odor):
        print("I feel some smell! Value:", odor)
        self.last_state = 'smelling'
        self.choose_run_length(odor=odor)

    def release_pheromone(self):
        print('releasing pheromone...')
        self.environment.add_pheromone(self.coord_phi, lifetime=200)
