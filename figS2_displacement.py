import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from figurefirst import FigureLayout
import figurefirst as fifi
import pickle
import matplotlib as mpl
import time

#from plotting.fig1 import plot_traj
from fig2_displacement import plot_traj
from plotting_helpers import my_gray_colormap
from arena import load_arena_pickle

mpl.rc('font', size=7)
#with open('../analysis/configs/big_arena_painted.pickle', 'rb') as f:

ARENA = load_arena_pickle("data/big_arena_fr_black_shadow.pickle")

segments_mapping = {'baseline': 'pre',
                    'stimulation': 'stim',
                    'relocation': 'reloc',
                    'test_before_movement': 'poststim',
                    'after_relocation': 'test'}

# plot_kwargs = {
#     'x': dict(marker='*', linestyle='', color='blue', label='x-axis'),
#     'y': dict(marker='o', linestyle='', color='green', label='y-axis')
# }

_config = {
    'temperature':
        {'points': {
            'x': dict(marker='s', linestyle='', color='k', label='x-axis'),
            'y': dict(marker='o', linestyle='', color='gray', label='y-axis')
        },
            'estimation': {
                'x': dict(color='k'),
                'y': dict(color='gray'),
            }},

    'trajectories': {
        'df': "data/all_ds_t01_cm.csv.gz",
        'rewarded_ids': [6, 10, 28],
        'nonrewarded_ids': [15, 19, 32],
        # 'cmap': 'winter'
        'cmap': my_gray_colormap(0.25, 1.0)
    }
}
#  all rewarded flies: [ 0  1  2  3  4  5  6  9 10 11 12 16 20 23 25 28 29 30 33 34]
#  non-rewarded: [ 7  8 13 14 15 17 18 19 21 22 24 26 27 31 32 35 36 37 38 39]


def plot_temperature_profile(data_fname, layout, fig, **kwargs):
    meas_kwargs = kwargs['points']
    est_kwargs = kwargs['estimation']

    tmps = pd.read_csv(data_fname, '\t')
    ax_temperature = layout.axes[(fig, 'ax_temperature')]

    for axis, variable in zip(['x', 'y'], ['y', 'x']):
        df_along_ax = tmps[tmps.axis == axis]
        ax_temperature.plot(df_along_ax[variable], df_along_ax.t, **(meas_kwargs[axis]))

        t_estimate = np.polyfit(df_along_ax[variable], df_along_ax.t, 2)
        poly = np.poly1d(t_estimate)
        var_vals = np.linspace(-30, 30, 60)

        ax_temperature.plot(var_vals, poly(var_vals), linestyle='-', **(est_kwargs[axis]))

        print(variable, ':\n', np.poly1d(poly, variable=variable))
    ax_temperature.legend()
    # ty_estimate = np.polyfit(last_measurements_y.x, last_measurements_y.t, 2)
    #
    # py = np.poly1d(ty_estimate)
    #
    # yp = np.linspace(-30, 30, 60)

    # layout.insert_figures('target_layer_name')
    layout.append_figure_to_layer(layout.figures[fig], 'mpl_layer')


def load_trajectories(fname, ids=None, test_period_sec=None):
    alltraj = pd.read_csv(fname)
    if ids is not None:
        alltraj = alltraj[alltraj.fly.isin(ids)]
    if test_period_sec is not None:
        test_exclude_idx = alltraj[(alltraj.segment == 'after_relocation') & (alltraj.tseg > test_period_sec)].index
        alltraj = alltraj.drop(alltraj.index.intersection(test_exclude_idx))
    return alltraj


def plot_traj_examples(alltraj, fly_ids, layout, fig, test_period_sec=100, **kwargs):
    clrs = None
    for i, flyid in enumerate(fly_ids):
        for segment, ax_name in segments_mapping.items():
            kw = kwargs.copy()
            ax = layout.axes[(fig, '{}{}'.format(ax_name, i+1))]
            df_fly = alltraj[(alltraj.fly == flyid) & (alltraj.segment == segment)]
            if segment == 'after_relocation':
                df_fly = df_fly[df_fly.tseg <= test_period_sec]  # plot only 100 sec after relocation as test period
                kw['fictive_reward'] = (df_fly.iloc[0].estimated_food_x, df_fly.iloc[0].estimated_food_y)  # plot fictive reward

            clrs = plot_traj(ax, df_fly.x_px, df_fly.y_px, **kw)

    if (fig, 'ax_colorbar') in layout.axes:
        cbax = layout.axes[(fig, 'ax_colorbar')]
        f = layout.figures[fig]
        cbar = f.colorbar(clrs, cax=cbax, orientation='horizontal')
        cbar.set_ticks([0, 1])
    layout.append_figure_to_layer(layout.figures[fig], 'mpl_layer')

    return clrs


if __name__ == '__main__':
    # todo: move parameters to config:
    # layout_fname,
    # flyids(3x),
    # fig_filename,
    # data_source (should contain flyid!),
    # test_period length (not implemented yet)
    # arena?

    layout_fname = 'fig_layouts/layout_s1.svg'
    fig_fname = 'figures/S2_displacement.svg'

    t_data_fname = "data/temperature.tsv" # temperature_last_mes.tsv
    if not os.path.isfile(t_data_fname):
        raise FileNotFoundError(t_data_fname)

    layout = FigureLayout(layout_fname, autogenlayers=True, make_mplfigures=True, hide_layers=['fifi_axs'])
    print(layout.figures)
    plot_temperature_profile(t_data_fname, layout, fig='fig_temperature', **(_config['temperature']))

    rew_ids = _config['trajectories']['rewarded_ids']
    nonrew_ids = _config['trajectories']['nonrewarded_ids']
    alltraj = load_trajectories(_config['trajectories']['df'], ids=rew_ids + nonrew_ids)
    plot_traj_examples(alltraj, rew_ids, layout, fig='fig_traj_rew', cmap=_config['trajectories']['cmap'], arena=ARENA)
    plot_traj_examples(alltraj, nonrew_ids, layout, fig='fig_traj_nonrew', cmap=_config['trajectories']['cmap'], arena=ARENA)

    #**(_config['trajectories']))

    layout.write_svg(fig_fname)

# temperature profile fitting:
# y :
#           2
# 0.01424 y - 0.00317 y + 23.15
# x :
#           2
# 0.01339 x - 0.03681 x + 21.8