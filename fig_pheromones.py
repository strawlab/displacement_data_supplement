import os

import numpy as np
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import figurefirst as fifi
from figurefirst import FigureLayout

from arena import plot_arena_histogram, load_arena_pickle

mpl.rc('font', size=6)


def kill_spines(ax):
    return fifi.mpl_functions.adjust_spines(ax, 'none',
                                            spine_locations={},
                                            smart_bounds=True,
                                            xticks=None,
                                            yticks=None,
                                            linewidth=1)


def plot_hist2d_from_npz(npz_filename, ax, arena, vmax=None):
    npzfile = np.load(npz_filename)
    hvalues = npzfile['h']
    xedges = npzfile['xe']
    yedges = npzfile['ye']
    im = plot_arena_histogram(hvalues, xedges, yedges, ax, arena, vmax=vmax, labeled_cbar=False)
    return im


_config = {'heatmap_max': 1.2,
           'fractions': {'colors': {'mass_test_no': 'gray', 'mass_test': 'red'},
                         'orient': 'v'}
           }


def heatmaps_figure(layout):
    ax_emitter = layout.axes[('fig_heatmaps', 'ax_heatmap_emitter')]
    ax_no = layout.axes[('fig_heatmaps', 'ax_heatmap_no')]
    ax_colorbar = layout.axes[('fig_heatmaps', 'ax_colorbar')]
    heatmap_emitter = "data/pheromones/heatmap_emitter.npz"
    heatmap_no = "data/pheromones/heatmap_no_emitter.npz"
    if not os.path.isfile(heatmap_emitter):
        raise FileNotFoundError(heatmap_emitter)
    if not os.path.isfile(heatmap_emitter):
        raise FileNotFoundError(heatmap_no)
    plot_hist2d_from_npz(heatmap_emitter, ax_emitter, arena, vmax=_config['heatmap_max'])
    clrs_heatmap = plot_hist2d_from_npz(heatmap_no, ax_no, arena, vmax=_config['heatmap_max'])
    f = layout.figures['fig_heatmaps']
    cbar = f.colorbar(clrs_heatmap, cax=ax_colorbar, orientation='vertical')
    cbar.set_ticks([0, _config['heatmap_max']])
    fifi.mpl_functions.adjust_spines(ax_colorbar, ['right'],
                                     spine_locations={'right': 0}, spine_location_offset=0,
                                     direction='out',
                                     smart_bounds=False, tick_length=2)
    layout.append_figure_to_layer(layout.figures['fig_heatmaps'], 'fig_heatmaps', cleartarget=True)


def fractions_figure(layout, config):
    colors_cfg = config['colors']
    orient = config['orient']
    print(orient)

    ax_fractions = layout.axes[('fig_fractions', 'ax_fractions')]

    fname_fractions = "data/pheromones/pheromone_test_fracs5_30.tsv"

    df_fracs = pd.read_csv(fname_fractions, sep='\t')
    swarm_palette = [colors_cfg['mass_test_no'], colors_cfg['mass_test']]
    print(df_fracs[['experiment', 'fraction']].head())
    # f, ax = plt.subplots(figsize=(2, 3))

    # kill_spines(ax_fractions)

    if orient == 'h':
        sns.swarmplot(y='experiment', x='fraction', data=df_fracs,
                      order=['mass_test_no', 'mass_test'], palette=swarm_palette, size=3,
                      ax=ax_fractions)
        splot = sns.boxplot(y='experiment', x='fraction', data=df_fracs,
                            order=['mass_test_no', 'mass_test'],
                            showcaps=False, boxprops={'facecolor': 'None'},
                            showfliers=False, whiskerprops={'linewidth': 0}, ax=ax_fractions)
    else:
        splot = sns.swarmplot(x='experiment', y='fraction', data=df_fracs, hue='experiment',
                      order=['mass_test_no', 'mass_test'], palette=swarm_palette, size=3,
                      ax=ax_fractions["axis"])
        splot = sns.boxplot(x='experiment', y='fraction', data=df_fracs,
                            order=['mass_test_no', 'mass_test'],
                            showcaps=False, boxprops={'facecolor': 'None'},
                            showfliers=False, whiskerprops={'linewidth': 0}, ax=ax_fractions)
        fifi.mpl_functions.adjust_spines(ax_fractions, ['left', 'bottom'],
                                         spine_locations={'left': 0, 'bottom': 0}, spine_location_offset=0,
                                         yticks=[0, 0.04, 0.08, 0.12], direction='out',
                                         smart_bounds=False, tick_length=2)
        plt.legend([], [], frameon=False)

        for tl in ax_fractions.get_xticklabels():
            tl.set_visible(False)

    ax_fractions.set_ylabel('')
    ax_fractions.set_xlabel('')
    layout.append_figure_to_layer(layout.figures['fig_fractions'], 'fig_fractions', cleartarget=True)


if __name__ == '__main__':
    settings = dict(layout_fname='fig_layouts/layout_pheromones.svg',
                                fig_fname='figures/fig_pheromones.svg',
                                arena_fname="data/pheromones_mass_arena.pickle")

    arena = load_arena_pickle(settings['arena_fname'])

    layout = FigureLayout(settings['layout_fname'], autogenlayers=True, make_mplfigures=True,
                          hide_layers=['layer_fifi_labels'])
    print(layout.axes)

    heatmaps_figure(layout)
    fractions_figure(layout, _config['fractions'])

    layout.write_svg(settings['fig_fname'])
    # t-test:
    #  ttest_ind(meta_df[meta_df.experiment=='mass_test'].fraction,
    #            meta_df[meta_df.experiment=='mass_test_no'].fraction,equal_var=False)
