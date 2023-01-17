import os

import pandas as pd
import numpy as np

import matplotlib as mpl
from figurefirst import FigureLayout
import figurefirst as fifi
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rc('font', size=7)
mpl.rc('svg', fonttype='none')
# plt.rcParams['text.usetex'] = True


_config_ = dict(circling=dict(folder="/mnt/strawscience/anna/code/behbahani_model/data/circling/",
                              fname_prefix='circling_simulations5',
                              example_flyid=42,
                              nbins=111
                              ),
                foods3=dict(nbins=25,
                            folder="/mnt/strawscience/anna/code/behbahani_model/data/rewards3/",
                            examples_file="three_rewards_5examples.csv",
                            spines={'traj_top': ['left'],
                                    'traj_middle': ['left'],
                                    'traj_bottom': ['left', 'bottom']},
                            post_ap_runs_file="three_rewards_5_postAP_runs_selected.csv",
                            hists_kw={'top': {'linestyle': '-', 'linewidth': 1, 'color': 'darkgreen'},
                                      'middle': {'linewidth': 1, 'color': 'navy'},
                                      'bottom': {'linewidth': 1, 'color': 'orchid'}
                                      }))


def plot_circling(layout: FigureLayout, fig: str, config: dict):
    # get figure and axes
    f = layout.figures[fig]
    ax_trajs = layout.axes[(fig, 'ax_trajs')]
    ax_hist = layout.axes[(fig, 'ax_distrib')]

    # load data
    df = pd.read_csv(os.path.join(config['folder'], config['fname_prefix'] + "_postAP_preprocessed.csv"))
    pd_runs = pd.read_csv(os.path.join(config['folder'], config['fname_prefix'] + "_post_return_runs.csv"))
    one_example = df[df.flyid == config['example_flyid']].copy()

    # plotting
    pos_scale = 2 * np.pi  # position in full revolutions
    t_scale = 2.  # time in seconds

    for i, data in one_example.groupby('iteration'):
        ax_trajs.plot(data.t_post / t_scale, data.relative_angle / pos_scale, lw=0.7)
        ax_trajs.plot(data[data.return_status == 'pre_return'].t_post / t_scale,
                   data[data.return_status == 'pre_return'].relative_angle / pos_scale, color='k', lw=0.7)
        ax_trajs.plot(data[data.eating].t_post / t_scale, data[data.eating].relative_angle / pos_scale, '.', color='red')
        ax_trajs.plot(data[data.smelling].t_post / t_scale, data[data.smelling].relative_angle / pos_scale, '.',
                   color='cyan', ms=3)

    yticks = np.arange(-5, 6)
    fifi.mpl_functions.adjust_spines(ax_trajs, ['left', 'bottom'],
                                     spine_locations={'left': 0, 'bottom': 0}, spine_location_offset=0,
                                     # xticks=[0, 20, 40],
                                     yticks=yticks,
                                     direction='out',
                                     smart_bounds=False, tick_length=2)

    ax_trajs.set_yticks(yticks)
    ax_trajs.grid(axis='y', color='0.95')
    ax_trajs.set_title("Trajectories in postAP for one simulation")
    ax_trajs.set_ylim(-5, 5)

    ax_trajs.set_xlabel('Time, sec')
    ax_trajs.set_ylabel('Position, revolutions')

    # histograms
    h = ax_hist.hist(pd_runs.run_midpoint / pos_scale, bins=config['nbins'],
                     orientation='horizontal', zorder=3, histtype='step', color='k')
    # yticks = np.arange(-5, 6)
    fifi.mpl_functions.adjust_spines(ax_hist, ['left','bottom'],
                                     spine_locations={'left': 0, 'bottom': 0}, spine_location_offset=0,
                                     # xticks=[0, 20, 40],
                                     yticks=yticks,
                                     direction='out',
                                     smart_bounds=False, tick_length=2)

    ax_hist.grid(axis='y', color='0.95', zorder=0)
    # ax_hist.grid(axis='y', color='0.95')
    ax_hist.set_title('Run midpoints distribution')
    #ax_hist.xlabel('Run midpoint, revolutions')
    ax_hist.set_xlabel('Count')
    ax_hist.set_ylim(-5, 5)


    # append figure
    layout.append_figure_to_layer(f, 'mpl_layer')


def plot_3foods(layout: FigureLayout, fig: str, config: dict):
    # get figure and axes
    f = layout.figures[fig]
    ax_traj_top = layout.axes[(fig, 'traj_top')]
    ax_traj_middle = layout.axes[(fig, 'traj_mid')]
    ax_traj_bottom = layout.axes[(fig, 'traj_bottom')]

    ax_hist = layout.axes[(fig, 'ax_midpoints')]

    axs_dict = {0: dict(name='middle', ax=ax_traj_middle),
                1: dict(name='top', ax=ax_traj_top),
                2: dict(name='bottom', ax=ax_traj_bottom)}

    # load data
    df = pd.read_csv(os.path.join(config['folder'], config['examples_file']))
    post_runs_good = pd.read_csv(os.path.join(config['folder'], config['post_ap_runs_file']))
    # plotting example trajectories
    tk = 2.  # step 0.5 sec, to convert time to seconds
    anglek = 2 * np.pi
    for flyid, data in df.groupby('flyid'):
        food_index = data.last_food_index.iloc[0]
        ax = axs_dict[food_index]["ax"]
        sregion = axs_dict[food_index]["name"]
        axname = f"traj_{sregion}"

        ax.plot(data.t / tk, data.angle / anglek, color='k')
        ax.plot(data[data.eating].t / tk, data[data.eating].angle / anglek, '.', color='red', ms=3)
        ax.plot(data[data.smelling].t / tk, data[data.smelling].angle / anglek, '.', color='cyan', ms=3)
        # ax.axhline(data.last_food_coord.iloc[0] / anglek, ls='--')
        # ax.axhline(data.last_food_coord.iloc[0] + 2 * np.pi, ls='--')
        # ax.axhline(data.last_food_coord.iloc[0] - 2 * np.pi, ls='--')
        ax.axvline(600 / tk, color='grey')
        # ax.set_ylabel('Postition, revolutions')
        fifi.mpl_functions.adjust_spines(ax, config['spines'][axname],
                                         spine_locations={'left': 0, 'bottom': 0}, spine_location_offset=0,
                                         # xticks=[0, 20, 40],
                                         yticks=[-0.5, 0, 0.5],
                                         direction='out',
                                         smart_bounds=False, tick_length=2)
        ax.set_xlim(200, 450)
        ax.set_ylim(-0.5, 0.65)

    # plotting run lenghts histogram

    # sns does not work with figurefirst: AttributeError: 'Axes' object has no attribute 'add_legend'
    # sns.histplot(data=post_runs_good, x='theta_midpoint', hue='last_food_index',
    #              element='step', fill=False, stat='density', common_norm=False, bins=config['nbins'], ax=ax_hist)
    for last_foodi in [1, 0, 2]:
        data = post_runs_good[post_runs_good.last_food_index == last_foodi].theta_midpoint
        food_name = axs_dict[last_foodi]["name"]
        my_kw = config['hists_kw'][food_name]
        ax_hist.hist(data, bins=config['nbins'], density=True, histtype='step', label=food_name, **my_kw)
    ax_hist.set_title("Distribution of run midpoints")
    # ax_hist.legend()
    fifi.mpl_functions.adjust_spines(ax_hist, ['left', 'bottom'],
                                     spine_locations={'left': 0, 'bottom': 0}, spine_location_offset=0,
                                     # xticks=[0, 20, 40],
                                     xticks=[-np.pi, 0, np.pi],
                                     direction='out',
                                     smart_bounds=False, tick_length=2)
    # removed because pi was not looking good, inserted in inkscape
    # ax_hist.set_xticklabels([r'- $\pi$', '0', r'$\pi$'])
    ax_hist.set_xticklabels([])

    # append figure
    layout.append_figure_to_layer(f, 'mpl_layer')


if __name__ == '__main__':
    layout_fname = 'layouts/ph_model.svg'
    fig_fname = 'output/figS_ph_model.svg'

    layout = FigureLayout(layout_fname, autogenlayers=True, make_mplfigures=True,
                          hide_layers=['fifi_axs', 'old_plots'])
    print(layout.figures)

    plot_circling(layout, 'fig_circling', _config_['circling'])
    plot_3foods(layout, 'fig_3foods', _config_['foods3'])

    layout.write_svg(fig_fname)
