import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from figurefirst import FigureLayout
import figurefirst as fifi
from arena import arena_hist2d, my_hist2d, plot_trajectory, load_arena_pickle
import pickle
import matplotlib as mpl
import seaborn as sns

from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

# plt.style.use('seaborn-paper')
# from plotting.figS2 import mypolarhist, plot_arcs
from plotting_helpers import my_gray_colormap, my_arrow, make_legend_arrow, conditions_mapping, mypolarhist, plot_arcs

mpl.rc('font', size=7)
mpl.rc('svg', fonttype='none')
# sns.set_style("whitegrid")


ARENA = load_arena_pickle('data/big_arena_fr_black_shadow.pickle')

segments_mapping = {'baseline': 'baseline',
                    'stimulation': 'stim',
                    'relocation': 'reloc',
                    'test_before_movement': 'poststim',
                    'after_relocation': 'test'}

_condition_colors = {'rewarded': 'red',
                     'non-rewarded': 'blue'}
_scatter_params_common = dict(s=11, alpha=0.5)
_scatter_params_condition = {'rewarded': dict(marker='x'),
                             'non-rewarded': dict(facecolors='none', s=15)}


def get_scatter_kw(condition):
    kw = _scatter_params_common.copy()
    kw.update(_scatter_params_condition[condition])
    kw.update(dict(color=_condition_colors[condition], label=condition))

    return kw


segments_pre = ['baseline', 'stimulation', 'test_before_movement']

# todo: move to config
# _color_fictive = 'blue'
# _color_actual = 'red'

# _clr_vav = '#4a008c'
_clr_vav = 'k'  # color of vector average for starting trajectories
_clr_reloc = 'limegreen'  # color of relocation vector direction
_direction_len = 3

_color_fictive = 'darkorange'
_color_actual = 'red'
_dists_kw = {'actual': {'linestyle': '-', 'linewidth': 1, 'color': _color_actual},
             'fictive': {'linewidth': 2, 'color': _color_fictive}}
# 'linestyle': (0,(2,2)) # dashes, see https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html


_config_ = dict(nbins=20, traj_start_sec=5, pre_vmax=15, post_vmax=3,
                cmap_hist2d='viridis', n_bins_scatter_hist=7,
                directions=dict(test_start='data/stats/after_reloc_state.tsv',
                                vectors='data/stats/start_vectors.tsv',
                                rmax=11,
                                arc=dict(lw=2, r_fr=9.5, r_ar=10.5),
                                polarhist_kw=dict(rscatter=8, ticklabels=False, grid=True,
                                                  scatter_size=6, scatter_alpha=0.4)
                                ))

# _config_ = dict(nbins=20, traj_start_sec=5, pre_vmax=15, post_vmax=3,
#                 cmap_hist2d=my_gray_colormap(0.25, 1.), n_bins_scatter_hist=7)


def plot_hist(ax, x, y, **kwargs):
    arena = kwargs.get('arena', ARENA)
    logscale = kwargs.get('logscale', False)
    vmin = kwargs.get('vmin', 0)
    vmax = kwargs.get('vmax', None)
    cmap = kwargs.get('cmap', None)
    # print('plot_hist: vmin {}, vmax {}, x {}'.format(vmin, vmax, x.shape))
    clrs = arena_hist2d(x, y, ax, arena, logscale=logscale,
                        labeled_cbar=False, frame_visible=False,
                        vmin=vmin, vmax=vmax, cmap=cmap)
    # print(clrs)
    arena.objects['fictive_reward']['visible'] = False
    arena.objects['fictive_reward_shadow']['visible'] = False
    return clrs


def plot_walking_hists_noshift(layout, alltraj, rz_fname, arena=ARENA, test_period_sec=100, nbins=30, bintype='square',
                               pre_vmax=5, post_vmax=2, **kwargs):
    # alltraj = pd.read_csv(alltraj_fname)
    print('flies:', alltraj.fly.unique())
    test_exclude_idx = alltraj[(alltraj.segment == 'after_relocation') & (alltraj.tseg > test_period_sec)].index
    alltraj = alltraj.drop(alltraj.index.intersection(test_exclude_idx))

    df_test = alltraj[alltraj.segment == 'after_relocation']
    test_shifted_x_cm = (df_test.x_px - df_test.estimated_food_x) / arena.px_to_cm_ratio
    test_shifted_y_cm = (df_test.y_px - df_test.estimated_food_y) / arena.px_to_cm_ratio

    # set fictive reward coordinates to mean frz
    meanrz = pd.read_csv(rz_fname)
    meanrz.set_index('condition', inplace=True)

    cmap = 'jet'
    clrs_pre = None
    clrs_post = None
    if bintype == 'square':
        arena.xy_binning(nbins, nbins)
        arena.objects['bins']['visible'] = False

        for condition, ax_condition in conditions_mapping.items():

            for segment in segments_pre:
                ax_segm = segments_mapping[segment]
                # ax = layout.axes['{}_{}'.format(ax_segm, ax_condition)]['axis']
                ax = layout.axes[('fig_heatmaps_pre', '{}_{}'.format(ax_segm, ax_condition))]
                df = alltraj[(alltraj.segment == segment) & (alltraj.condition == condition)]
                print(condition, segment, df.shape)
                clrs_pre = plot_hist(ax, df.x_px, df.y_px, arena=arena, cmap=cmap, vmax=pre_vmax)
                print(condition, segment, clrs_pre.get_array().max())

            for objname in ['fictive_reward_shadow', 'fictive_reward']:
                arena.objects[objname]['x'] = meanrz.loc[condition, 'fictive_reward_x_px']
                arena.objects[objname]['y'] = meanrz.loc[condition, 'fictive_reward_y_px']
                arena.objects[objname]['visible'] = True

            df = alltraj[(alltraj.segment == 'after_relocation') & (alltraj.condition == condition)]
            # testax = layout.axes['test_{}'.format(ax_condition)]['axis']
            testax = layout.axes['fig_heatmaps_post', 'test_{}'.format(ax_condition)]

            clrs_post = plot_hist(testax, df.x_px, df.y_px, arena=arena, cmap=cmap, vmax=post_vmax)
            print('MAX', condition, 'test', clrs_post.get_array().max())

    # cbar_ax = layout.axes['colorbar_stim']['axis']
    cbar_ax = layout.axes[('fig_heatmaps_pre', 'colorbar_stim')]
    f = layout.figures['fig_heatmaps_pre']
    cbar = f.colorbar(clrs_pre, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, pre_vmax])

    # cbar_ax = layout.axes['colorbar_test']['axis']
    ftest = layout.figures['fig_heatmaps_post']
    cbar_ax = layout.axes[('fig_heatmaps_post', 'colorbar_test')]
    cbar = ftest.colorbar(clrs_post, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, post_vmax])


def plot_walking_hists_all(layout, alltraj, rz_fname, arena=ARENA, test_period_sec=100, nbins=30, bintype='square',
                           pre_vmax=5, post_vmax=2, **kwargs):
    print('flies:', alltraj.fly.unique())
    test_exclude_idx = alltraj[(alltraj.segment == 'after_relocation') & (alltraj.tseg > test_period_sec)].index
    alltraj = alltraj.drop(alltraj.index.intersection(test_exclude_idx))

    df_test = alltraj[alltraj.segment == 'after_relocation']

    cmap = kwargs.get('cmap_hist2d', 'jet')
    clrs_pre = None
    if bintype == 'square':
        arena.xy_binning(nbins, nbins)
        arena.objects['bins']['visible'] = False

        for condition, ax_condition in conditions_mapping.items():
            for segment in segments_pre:
                ax_segm = segments_mapping[segment]
                # ax = layout.axes['{}_{}'.format(ax_segm, ax_condition)]['axis']
                ax = layout.axes[('fig_heatmaps_pre', '{}_{}'.format(ax_segm, ax_condition))]
                df = alltraj[(alltraj.segment == segment) & (alltraj.condition == condition)]
                print(condition, segment, df.shape)
                clrs_pre = plot_hist(ax, df.x_px, df.y_px, arena=arena, cmap=cmap, vmax=pre_vmax)
                print(condition, segment, clrs_pre.get_array().max())

    clrs_post = plot_test_hists_shift(layout, df_test, rz_fname, arena, nbins, post_vmax=post_vmax, cmap=cmap)

    cbar_ax = layout.axes[('fig_heatmaps_pre', 'colorbar_stim')]
    f = layout.figures['fig_heatmaps_pre']
    cbar = f.colorbar(clrs_pre, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, pre_vmax])

    # cbar_ax = layout.axes['colorbar_test']['axis']
    ftest = layout.figures['fig_heatmaps_post']
    cbar_ax = layout.axes[('fig_heatmaps_post', 'colorbar_test')]
    cbar = ftest.colorbar(clrs_post, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, post_vmax])


def plot_test_hists_shift(layout, df_test, rz_fname, arena=ARENA, nbins=30, bintype='square',
                          post_vmax=5, cmap='jet', **kwargs):
    print('shifted test hist2d params:', nbins, post_vmax)
    test_shifted_x_cm = (df_test.x_px - df_test.estimated_food_x) / arena.px_to_cm_ratio
    test_shifted_y_cm = (df_test.y_px - df_test.estimated_food_y) / arena.px_to_cm_ratio

    print(arena.objects)

    # set actual reward coordinates to mean rz
    meanrz = pd.read_csv(rz_fname)
    meanrz.set_index('condition', inplace=True)

    clrs_post = None
    # arena.xy_binning(nbins, nbins)
    # arena.objects['bins']['visible'] = False

    rew_radius_cm = arena.objects['reward']['radius'] / arena.px_to_cm_ratio
    print('reward radius: {}'.format(rew_radius_cm))

    # binning
    arena.xy_binning(nbins, nbins)
    arena.objects['bins']['visible'] = False
    xbins = arena.objects['bins']['xbins']
    ybins = arena.objects['bins']['ybins']
    binsizex = xbins[1] - xbins[0]
    binsizey = ybins[1] - ybins[0]

    binsizex_cm = binsizex / arena.px_to_cm_ratio
    binsizey_cm = binsizey / arena.px_to_cm_ratio
    print("bins [cm]:", binsizex_cm, binsizey_cm)
    xmin = test_shifted_x_cm.min()
    ymin = test_shifted_y_cm.min()
    cmbinsx = np.arange(xmin, xmin + binsizex_cm * arena.get_nbins() + 1, binsizex_cm)
    cmbinsy = np.arange(ymin, ymin + binsizey_cm * arena.get_nbins() + 1, binsizey_cm)
    print(cmbinsx.min(), cmbinsx.max())

    for condition, ax_condition in conditions_mapping.items():
        xrew = meanrz.loc[condition, 'reward_relative_x']
        yrew = meanrz.loc[condition, 'reward_relative_y']

        df = df_test[df_test.condition == condition]

        shiftestax = layout.axes['fig_heatmaps_post', 'test_{}'.format(ax_condition)]

        fictive_reward_location = plt.Circle((0, 0), radius=rew_radius_cm,
                                             **arena.objects['fictive_reward']['plot_kwargs'])
        fictive_reward_location_shadow = plt.Circle((0, 0), radius=rew_radius_cm,
                                                    **arena.objects['fictive_reward_shadow']['plot_kwargs'])
        actual_rz = plt.Circle((xrew, yrew), radius=rew_radius_cm,
                               color='none', ec='red', zorder=5)

        shiftestax.add_artist(fictive_reward_location)
        shiftestax.add_artist(fictive_reward_location_shadow)
        shiftestax.add_artist(actual_rz)

        clrs_shifted = my_hist2d(test_shifted_x_cm.loc[df.index], test_shifted_y_cm.loc[df.index],
                                 shiftestax, xbins=cmbinsx, ybins=cmbinsy,
                                 show_cbar=False, vmin=0, vmax=post_vmax, cmap=cmap, axes_visible=False)
        print(condition, 'test(FICTIVE)', clrs_shifted.get_array().max())

    return clrs_shifted


def plot_fractions_scatter(layout, stats_fname, **kwargs):
    stats = pd.read_csv(stats_fname, sep='\t')
    n_bins = kwargs.get('n_bins_scatter_hist', 5)

    # tte = [rew_color, nonrew_color]
    ax_sc = layout.axes[('fig_fractions', 'ax_scatter_fracs')]
    ax_hist = layout.axes[('fig_fractions', 'ax_diffhist')]

    maxval = 0.4
    minval = -0.02
    ax_sc.set_xlim(minval, maxval)
    ax_sc.set_ylim(minval, maxval)

    ax_sc.plot([minval, 0.2], [minval, 0.2], '--k')
    ax_hist.axvline(0, ls='--', color='k')

    ticks = np.arange(0, 0.5, step=0.2)

    fifi.mpl_functions.adjust_spines(ax_sc, ['left', 'bottom'],
                                     spine_locations={'left': 0, 'bottom': 0}, spine_location_offset=0,
                                     xticks=ticks, yticks=ticks, direction='out',
                                     smart_bounds=False, tick_length=2)

    hbins = np.linspace(-0.25, 0.25, n_bins + 1)
    hticks = np.linspace(-0.3, 0.3, 3)
    ax_hist.set_xlim(-0.3, 0.3)
    fifi.mpl_functions.adjust_spines(ax_hist, ['left','bottom'],
                                     spine_locations={'bottom': 0, 'left': 0}, spine_location_offset=0,
                                     xticks=hticks, yticks=[0, 15], direction='out',
                                     smart_bounds=False, tick_length=1.5)

    sc = []
    for condition, cond_label in conditions_mapping.items():
        stats_loc = stats[(stats.condition == condition) & (stats.segment == 'test100')]
        plot_kw = get_scatter_kw(condition)
        sc.append(ax_sc.scatter(stats_loc.at_fictive_reward, stats_loc.at_reward, **plot_kw))
        ax_hist.hist(stats_loc.at_fictive_reward - stats_loc.at_reward, bins=hbins, histtype='step',
                     color=_condition_colors[condition])
    ax_hist.set_ylabel('Count')
    # ax_hist.set_xlabel('Fictive - actual')
    ax_hist.set_xlabel('Difference')

    ax_leg = layout.axes[('fig_fractions', 'ax_sc_legend')]
    ax_leg.axis('off')
    ax_leg.legend(handles=sc, frameon=False)


def plot_fractions(layout, stats_fname, fr_color='blue', ar_color='red'):
    stats = pd.read_csv(stats_fname, sep='\t')
    swarm_palette = [fr_color, ar_color]

    for condition, cond_label in conditions_mapping.items():
        ax = layout.axes[("group_fraction", "ax_frac_{}".format(cond_label))]

        stats_loc = stats[(stats.condition == condition) & (stats.segment == 'test100')]
        # sl_nonrew = stats[(stats.condition == 'non-rewarded') & (stats.segment == 'test100')]

        sns.swarmplot(x='location', y='fraction', data=stats_loc,
                      order=['at_fictive_reward', 'at_reward'], palette=swarm_palette, size=3,
                      ax=ax)
        sns.boxplot(x='location', y='fraction', data=stats_loc, order=['at_fictive_reward', 'at_reward'],
                    showcaps=False, boxprops={'facecolor': 'None'},
                    showfliers=False, whiskerprops={'linewidth': 0}, ax=ax)
        # ax.axis('off')
        fifi.mpl_functions.adjust_spines(ax, 'left',
                                         spine_locations={'left': 0}, spine_location_offset=0,
                                         yticks=[0, 0.1, 0.2], direction='out', smart_bounds=False, tick_length=2)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['left'].set_bounds(0, 0.2)

    ax_nr = layout.axes[("group_fraction", "ax_frac_nonrew")]
    ax_nr.set_ylim(-0.02, 0.2)
    ax_r = layout.axes[("group_fraction", "ax_frac_rew")]
    ax_r.set_ylim(-0.02, 0.35)
    # ax_r.set_aspect(ax_nr.get_aspect())


def plot_dists(layout, alltraj, test_sec=100, binsize_cm=2, fr_color='blue', ar_color='red'):
    df_test100 = alltraj[(alltraj.segment == 'after_relocation') & (alltraj.tseg <= test_sec)]
    bins = np.arange(0, 40, binsize_cm)

    spines = {'ax_dists_rew': ['left'],
              'ax_dists_nonrew': ['left', 'bottom']}

    # kw = {'actual': {'linestyle': '-', 'linewidth': 1, 'color': ar_color},
    #       # 'fictive': {'linestyle': '-', 'linewidth': 3, 'color': 'darkorange'}}
    #       'fictive': {'linestyle': '-', 'linewidth': 2, 'color': fr_color}}

    kw = _dists_kw

    for condition, cond_label in conditions_mapping.items():
        axname = "ax_dists_{}".format(cond_label)
        ax = layout.axes[("fig_dist_distrib", axname)]
        df = df_test100[df_test100.condition == condition]
        # ax.hist(df.dist_fictive_reward_cm, bins=bins, density=True, histtype='step', label='fictive', color='grey', lw=2)
        ax.hist(df.dist_fictive_reward_cm, bins=bins, density=True, histtype='step', label='fictive', **(kw['fictive']))
        ax.hist(df.distance_reward_cm, bins=bins, density=True, histtype='step', label='actual', **(kw['actual']))

        fifi.mpl_functions.adjust_spines(ax, spines[axname],
                                         spine_locations={'left': 0, 'bottom': 0}, spine_location_offset=0,
                                         xticks=[0, 20, 40], yticks=[0, 0.07], direction='out',
                                         smart_bounds=False, tick_length=2)

        ax.set_xlim(0, 45)
        ax.set_ylim(0, 0.075)

    # ax = layout.axes[("fig_dist_distrib", "ax_dists_nonrew")]
    # ax.legend()


def get_average_vector(xs, ys, normalize_len=None):
    average_vector_x = xs.mean()
    average_vector_y = ys.mean()
    if normalize_len is None:
        return average_vector_x, average_vector_y
    res_x = average_vector_x / np.sqrt(average_vector_x ** 2 + average_vector_y ** 2) * normalize_len
    res_y = average_vector_y / np.sqrt(average_vector_x ** 2 + average_vector_y ** 2) * normalize_len
    return res_x, res_y


def plot_traj_starts(layout, fname, traj_start_sec, **kwargs):
    df = pd.read_csv(fname)
    df_starts = df[(df.segment == 'after_relocation') & (df.tseg <= traj_start_sec)]
    vav = None
    gray_cmp = my_gray_colormap()
    direction_vec_length = 5

    for condition, cond_label in conditions_mapping.items():
        axname = "ax_start_{}".format(cond_label)
        ax = layout.axes[("fig_trajs", axname)]
        df_condition = df_starts[df_starts.condition == condition]

        for fly, df_fly in df_condition.groupby('fly'):
            x = df_fly.rel_x_cm
            y = df_fly.rel_y_cm
            clrs = plot_trajectory(x, y, ax, markersize=5, tmax=traj_start_sec, cmap=gray_cmp)  # , rasterized=False)

        last_points = df_condition.groupby('fly').last()
        first_points = df_condition.groupby('fly').first()
        av_vec = get_average_vector(last_points.rel_x_cm, last_points.rel_y_cm)

        vav, = ax.plot([0, av_vec[0]], [0, av_vec[1]], color=_clr_vav, lw=2.5, label='vector average',
                       zorder=6)

        fictive_reward_x = first_points['estimated_food_x_cm'] - first_points['x_cm']
        fictive_reward_y = first_points['estimated_food_y_cm'] - first_points['y_cm']
        fr_mean_direction = get_average_vector(fictive_reward_x, fictive_reward_y, normalize_len=_direction_len)
        # vav_fr, = ax.plot([0, fr_mean_direction[0]], [0, fr_mean_direction[1]], color='darkorange', lw=2,
        #                   label='mean fictive RZ\ndirection', zorder=5)

        relocation_mean_direction = get_average_vector(last_points.transx_cm, last_points.transy_cm,
                                                       normalize_len=_direction_len)
        ar_reloc = my_arrow(ax, -relocation_mean_direction[0], -relocation_mean_direction[1],
                            relocation_mean_direction[0], relocation_mean_direction[1],
                            linewidth=2, color=_clr_reloc, zorder=5, head_len=0.2)

        ar_fr = my_arrow(ax, 0, 0,
                         fr_mean_direction[0], fr_mean_direction[1],
                         linewidth=2, color='darkorange', zorder=5, head_len=0.2)

        ax.set_ylim(-9, 9)
        ax.set_aspect('equal')
        # ax.set_yticks(np.arange(-10, 11, 5))
        # ax.set_xticks(np.arange(-5, 6, 5))
        # ax.grid()
        ax.axis('off')

    # ax = layout.axes['ax_legend']['axis']
    ax = layout.axes[('fig_trajs', 'ax_legend')]
    ax.axis('off')
    # ax.legend([vav], ['vector average'])
    # ax.legend(handles=[vav, vav_fr, vav_reloc], loc='lower center', frameon=False)

    #["vector average", 'mean relocation\ndirection', 'mean fictive RZ\ndirection']
    ax.legend([vav, ar_reloc, ar_fr], ["flies", 'displacement', 'fictive RZ'],
              loc='lower center', frameon=False,
              handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow)})

    # plots 5cm scale bar
    # ax = layout.axes[("fig_trajs", "ax_start_nonrew")]
    ax = layout.axes[("fig_trajs", "ax_start_rew")]
    ax.plot([-2.5, 2.5], [-7., -7.], color='black', lw=3)


def plot_directions_stats(layout, fig, config):
    names = dict(fictive=dict(ax='fr', col='FRZ'),
                 actual=dict(ax='ar', col='ARZ'),
                 relocation=dict(ax='reloc', col='reloc'),
                 rew=dict(ax='rew', df='rewarded'),
                 nonrew=dict(ax='nonrew', df='non-rewarded'))

    start_vectors = pd.read_csv(config['vectors'], sep='\t')
    df_test_start = pd.read_csv(config['test_start'], sep='\t')

    get_ax = lambda condition, poi: layout.axes[(fig, '{}_{}'.format(names[condition]['ax'], names[poi]['ax']))]
    get_col = lambda poi: 'angle_{}'.format(names[poi]['col'])

    poi = 'fictive'
    for condition in ['rew', 'nonrew']:
        ax = get_ax(condition, poi)
        col = get_col(poi)
        condname = names[condition]['df']
        mypolarhist(-start_vectors[start_vectors.condition == condname][col], ax,
                    **(config['polarhist_kw']))
        plot_arcs(ax, df_test_start[df_test_start.condition == condname], **(config['arc']))
        ax.set_ylim([0, config['rmax']])

    layout.append_figure_to_layer(layout.figures[fig], 'mpl_layer')


def figure2(alltraj_fname, stats_fname, test_period_fname, layout_fname, output_fname, rz_fname, **kwargs):
    plotfrac = kwargs.get('plot_fracs_func', plot_fractions)

    layout = FigureLayout(layout_fname, autogenlayers=True, make_mplfigures=True)
    # print(json.dumps(layout.figures, indent=2))

    alltraj = pd.read_csv(alltraj_fname)
    plot_walking_hists_all(layout, alltraj, rz_fname, **kwargs)
    plot_traj_starts(layout, test_period_fname, **kwargs)
    plotfrac(layout, stats_fname, **kwargs)

    plot_directions_stats(layout, 'fig_directions', kwargs['directions'])

    layout.save(output_fname)


if __name__ == '__main__':
    # todo: move parameters to config:
    # layout_fname,
    # fig_filename,
    # data_source (should contain flyid!),
    # test_period length
    # vmax_pre, vmax_post for the histograms
    # nbins
    # histogram type (not implemented) -- square / hex
    # arena?

    alltraj_fname = "data/all_ds_t01_d2_cm_no2.csv.gz"

    layout_fname = 'fig_layouts/layout_displ_analysis.svg'

    stats_fname = 'data/stats/relocation_stats_t01_no2.csv'
    fig_fname = 'figures/analysis.svg'
    fracplotfunc = plot_fractions_scatter

    # layout_fname = 'layouts/fig2_layout.svg'
    if not os.path.isfile(layout_fname):
        raise FileNotFoundError(layout_fname)

    testper_fname = "data/test_cm_dt01.csv.gz"
    mean_rz_fname = "data/reward_zones/mean_rewards_coords.csv"

    if not os.path.isfile(alltraj_fname):
        raise FileNotFoundError(alltraj_fname)

    print(ARENA.objects['fictive_reward_shadow']['plot_kwargs'])

    figure2(alltraj_fname, stats_fname,
            test_period_fname=testper_fname,
            layout_fname=layout_fname,
            output_fname=fig_fname,
            rz_fname=mean_rz_fname,
            plot_fracs_func=fracplotfunc,
            **_config_
            )

    #
    # figure2(alltraj_fname, stats_fname,
    #         test_period_fname=testper_fname,
    #         layout_fname=layout_fname,
    #         output_fname=fig_fname,
    #         rz_fname=mean_rz_fname,
    #         traj_start_sec=5,
    #         pre_vmax=7, post_vmax=2,
    #         nbins=30,
    #         plot_fracs_func=fracplotfunc)
