import os
import pandas as pd
import matplotlib as mpl
from figurefirst import FigureLayout
import numpy as np
from matplotlib.patches import Circle
import figurefirst as fifi
import seaborn as sns

from arena import load_arena_pickle, plot_trajectory, arena_hist2d
from fig_displ_analysis import plot_hist
from figS2_displacement import load_trajectories
from plotting_helpers import my_gray_colormap, plot_arc_polar, conditions_mapping

from plotting_helpers import mypolarhist, plot_arcs

mpl.rc('font', size=7)
mpl.rc('svg', fonttype='none')

gray_cmap = my_gray_colormap(0.25, 1.0)

_config_ = {
    # 'layout':
    # 'figure':
    'downsample': dict(dt=0.1, dd=5,
                       folder='data/flytrax20181204_170930',
                       file_original='clean_flytrax20181204_170930.csv',
                       file_dst='ds_t01_flytrax20181204_170930.csv',
                       file_dstd='ds_t01_d2_flytrax20181204_170930.csv',
                       cmap_heatmap='viridis',
                       # arena='../analysis/configs/big_arena_fr_black_shadow.pickle',
                       arena='data/big_arena_fr_black_shadow.pickle',
                       nbins=20,
                       traj_markersize=1,
                       cmap_traj=gray_cmap),

    'coords': dict(  # example_flies=[3, 30],  # uncomment this line if want to plot examples of coord transform
        # alltraj="/mnt/strawscience/anna/experiments/relocation/all_traj/all_ds_t01_d2_cm_no2.csv.gz",
        data_rz='data/reward_zones/reward_locations.csv',
        arena_frz='data/reward_zones/cmarena_rewarded_fictive_rewards.pickle',
        lw_meanrz=2,
        traj_markersize=1),

    'directions': dict(test_start='data/stats/after_reloc_state.tsv',
                       vectors='data/stats/start_vectors.tsv',
                       polarhist_kw=dict(rscatter=9, ticklabels=False, grid=True,
                                         scatter_size=4, scatter_alpha=0.4),
                       arc=dict(lw=2, r_fr=10, r_ar=11),
                       rmax=11.5,
                       ),
    'distance_rz': dict(kwargs={'actual': {'linestyle': '-', 'linewidth': 1, 'color': 'red'},
                                'fictive': {'linewidth': 2, 'color': 'darkorange'}},
                        xlim=[0, 45], ylim=[0, 0.075],
                        test_sec=100, binsize_cm=2, dist_max=40,
                        spines={'ax_dists_rew': ['left'],
                                'ax_dists_nonrew': ['left', 'bottom']}
                        # spines={'ax_dists_rew': ['left', 'bottom'],
                        #         'ax_dists_nonrew': ['bottom']}
                        ),
    'alltraj_fname': "data/all_ds_t01_d2_cm_no2.csv.gz",
    'test_sec': 100,
    'walking_hists': dict(arena='data/big_arena_fr_black_shadow.pickle',
                          rz_fname="data/reward_zones/mean_fictive_rz_coords.csv",
                          cmap='viridis',
                          nbins=20,
                          vmax=2.7
                          ),
    'enter_exit': dict(enters='data/stats/enters_2cm_walking.csv',
                       exits='data/stats/exits_2cm_walking.csv',
                       stats='data/stats/enter_exit_intersections.tsv',
                       threshold=0.25,
                       plot_scalebar=False,
                       ntrajs=6,
                       color_enter=gray_cmap.colors[0],
                       color_exit=gray_cmap.colors[-1],
                       traj_markersize=10)
}


def plot_dist_hists(layout, fig, config):
    alltraj = config['alltraj']
    df_test100 = alltraj[alltraj.segment == 'after_relocation']

    bins = np.arange(0, config['dist_max'], config['binsize_cm'])

    # kw = {'actual': {'linestyle': '-', 'linewidth': 1, 'color': ar_color},
    #       # 'fictive': {'linestyle': '-', 'linewidth': 3, 'color': 'darkorange'}}
    #       'fictive': {'linestyle': '-', 'linewidth': 2, 'color': fr_color}}

    kw = config['kwargs']

    for condition, cond_label in conditions_mapping.items():
        axname = "ax_dists_{}".format(cond_label)
        ax = layout.axes[(fig, axname)]
        df = df_test100[df_test100.condition == condition]
        # ax.hist(df.dist_fictive_reward_cm, bins=bins, density=True, histtype='step', label='fictive', color='grey', lw=2)
        ax.hist(df.dist_fictive_reward_cm, bins=bins, density=True, histtype='step', label='fictive', **(kw['fictive']))
        ax.hist(df.distance_reward_cm, bins=bins, density=True, histtype='step', label='actual', **(kw['actual']))

        fifi.mpl_functions.adjust_spines(ax, config['spines'][axname],
                                         spine_locations={'left': 0, 'bottom': 0}, spine_location_offset=0,
                                         xticks=[0, 20, 40], yticks=[0, 0.07], direction='out',
                                         smart_bounds=False, tick_length=2)

        ax.set_xlim(config['xlim'])
        ax.set_ylim(config['ylim'])

    layout.append_figure_to_layer(layout.figures[fig], 'mpl_layer')


def plot_downsampling(layout, fig, config):
    arena = load_arena_pickle(config['arena'])
    arena.xy_binning(config['nbins'])

    for obj in arena.objects.keys():
        arena.set_object_visibility(obj, False)

    f = layout.figures[fig]
    get_ax = lambda what, state: layout.axes[(fig, '{}_{}'.format(what, state))]
    for state in ['original', 'dst', 'dstd']:
        fname = os.path.join(config['folder'], config['file_{}'.format(state)])
        df = pd.read_csv(fname)
        ax_traj = get_ax('traj', state)
        ax_hist = get_ax('hist', state)
        ax_cb = get_ax('colorbar', state)

        plot_trajectory(df.x_px, df.y_px, ax_traj, markersize=config['traj_markersize'], cmap=config['cmap_traj'])
        arena.plot(ax_traj, axes_visible=False, margin=0.01, with_centers=False)

        clrs = arena_hist2d(df.x_px, df.y_px, ax_hist, arena, logscale=False,
                            labeled_cbar=False, frame_visible=False,
                            cmap=config['cmap_heatmap'])
        minval = clrs.get_array().min()
        maxval = clrs.get_array().max()

        cbar = f.colorbar(clrs, cax=ax_cb, ticks=[minval, maxval], orientation='horizontal', format='%.1f')

        # todo: ad cororbars

    layout.append_figure_to_layer(f, 'mpl_layer')


def plot_coord_transform(layout, fig, config):
    lw = config.get('lw_meanrz', 3)

    def plot_mean_arz(axs, conditions, lw_mean=lw, lw_all=0.5, flyids=None):
        for iax, condition in enumerate(conditions):
            fictive_reward = Circle((0, 0), r_cm, ec='orange', color='none', lw=lw_mean)
            axs[iax].add_artist(fictive_reward)
            df_rz = rz[rz.condition == condition]
            if flyids is not None:
                df_rz = df_rz[df_rz.fly.isin(flyids)]
            for i, row in df_rz.iterrows():
                cur_actual_reward = Circle((row.reward_relative_x, row.reward_relative_y), r_cm, ec='gray',
                                           color='none',
                                           lw=lw_all)
                axs[iax].add_artist(cur_actual_reward)

            mean_rew_x = mean_coords.loc[condition, 'reward_relative_x']
            mean_rew_y = mean_coords.loc[condition, 'reward_relative_y']
            mean_reward = Circle((mean_rew_x, mean_rew_y), r_cm, ec='red', color='none', lw=lw_mean)
            axs[iax].add_artist(mean_reward)

            axs[iax].set_xlim(-30, 30)
            axs[iax].set_ylim(-30, 30)
            axs[iax].axis('off')

    rz = pd.read_csv(config['data_rz'])
    mean_coords = rz.groupby('condition').mean().drop(columns=['fly'])
    print(mean_coords)

    arena_frz = load_arena_pickle(config['arena_frz'])
    arena_frz.objects['reward']['plot_kwargs']['linewidth'] = lw
    arena_frz.objects['mean_fictive_reward']['plot_kwargs']['linewidth'] = lw
    r_cm = arena_frz.objects['reward']['radius']

    ax_mean_arz = layout.axes[(fig, 'mean_arz')]
    ax_mean_frz = layout.axes[(fig, 'mean_frz')]
    arena_frz.plot(ax_mean_frz, with_centers=False, axes_visible=False, margin=0.02)
    plot_mean_arz([ax_mean_arz], ['rewarded'])

    flyids = config.get('example_flies', None)
    if flyids is not None:
        if 'alltraj_fname' in config:
            alltrajs = load_trajectories(config['alltraj_fname'], ids=flyids, test_period_sec=100)
        else:
            assert 'alltraj' in config
            alltrajs = config['alltraj']

        trajs = alltrajs[(alltrajs.segment.isin(['relocation', 'after_relocation'])) & (alltrajs.fly.isin(flyids))]

        ax_traj_a = layout.axes[(fig, 'traj_a')]
        ax_traj_f = layout.axes[(fig, 'traj_f')]

        colors = ['blue', 'green']
        for flyid, c in zip(flyids, colors):
            flytraj = trajs[trajs.fly == flyid]
            plot_trajectory(flytraj.x_cm, flytraj.y_cm, ax_traj_a, colorful=False,
                            color=c, markersize=config['traj_markersize'])
            plot_trajectory(flytraj.x_cm - flytraj.estimated_food_x_cm.iloc[-1],
                            flytraj.y_cm - flytraj.estimated_food_y_cm.iloc[-1],
                            ax_traj_f, colorful=False, color=c, markersize=config['traj_markersize'])

        flyid_objs = ['fr{}'.format(flyid) for flyid in flyids]
        arena_frz.objects['mean_fictive_reward']['plot_kwargs']['zorder'] = 2
        arena_frz.objects['reward']['plot_kwargs']['zorder'] = 2
        for objname in arena_frz.objects.keys():
            if objname.startswith('fr') and objname not in flyid_objs:
                # print(objname)
                arena_frz.set_object_visibility(objname, False)
        arena_frz.plot(ax_traj_a, with_centers=False, margin=0.02, axes_visible=False)
        plot_mean_arz([ax_traj_f], ['rewarded'], flyids=flyids)

    layout.append_figure_to_layer(layout.figures[fig], 'mpl_layer')


def plot_enter_exit_stats(layout, fig, config, addfig=False):
    # plot polar hist of enter-exit directions; not used
    axs = {'rewarded': layout.axes[(fig, 'rew_inout')],
           'non-rewarded': layout.axes[(fig, 'nonrew_inout')]}
    enter_exit = pd.read_csv(config['enter_exit'], sep='\t')

    for condition in ['rewarded', 'non-rewarded']:
        ax = axs[condition]

        mypolarhist(enter_exit[enter_exit.condition == condition]['angle_enter_exit'], ax,
                    **(config['polarhist_kw']))
        # plot_arcs(ax, df_test_start[df_test_start.condition == condname], align_to=col_align,
        #           **(config['arc']))
        ax.set_ylim([0, config['rmax']])

    if addfig:
        layout.append_figure_to_layer(layout.figures[fig], 'mpl_layer')


def plot_directions_stats(layout, fig, config):
    # todo: avoid copy (fig_reloc_analysis.py)
    names = dict(fictive=dict(ax='fr', col='FRZ', align='angle_fr'),
                 actual=dict(ax='ar', col='ARZ', align='angle_ar'),
                 relocation=dict(ax='reloc', col='reloc', align='anlge_reloc_start'),
                 rew=dict(ax='rew', df='rewarded'),
                 nonrew=dict(ax='nonrew', df='non-rewarded'))

    start_vectors = pd.read_csv(config['vectors'], sep='\t')
    # columns of start_vectors:
    # fly
    # anti_reloc - [x,y] - vector opposite to relocation
    # actual_reward_vec  - vector to actual reward
    # fictive_reward_vec - vector to fictive reward
    # fly_vec - vector where the fly went
    # condition - rew/non-rew
    # angle_FRZ - angle between fly vector and fictive rz vector
    # angle_ARZ - angle between fly vector and actual rz vector
    # angle_reloc - angle between fly vector and anti-reloc vector.
    df_test_start = pd.read_csv(config['test_start'], sep='\t')
    print(df_test_start.columns)

    get_ax = lambda condition, poi: layout.axes[(fig, '{}_{}'.format(names[condition]['ax'], names[poi]['ax']))]
    get_col = lambda poi: 'angle_{}'.format(names[poi]['col'])

    for poi in ['actual', 'relocation']:
        # for poi in ['fictive', 'actual', 'relocation']:
        for condition in ['rew', 'nonrew']:
            ax = get_ax(condition, poi)
            col = get_col(poi)
            col_align = names[poi]['align']
            condname = names[condition]['df']
            mypolarhist(-start_vectors[start_vectors.condition == condname][col], ax,
                        **(config['polarhist_kw']))
            plot_arcs(ax, df_test_start[df_test_start.condition == condname], align_to=col_align,
                      **(config['arc']))
            ax.set_ylim([0, config['rmax']])

    layout.append_figure_to_layer(layout.figures[fig], 'mpl_layer')


#
# def get_hists(df, column, groupby):
#

def plot_walking_hists(layout, fig, config):
    arena = load_arena_pickle(config['arena'])
    arena.xy_binning(config['nbins'])
    arena.objects['bins']['visible'] = False

    alltraj = config['alltraj']
    df_test = alltraj[alltraj.segment == 'after_relocation']

    cmap = config.get('cmap', 'viridis')
    post_vmax = config.get('vmax', None)

    # from here: copied from fig_reloc_analysis: plot_walking_hists_noshift

    meanrz = pd.read_csv(config['rz_fname'])
    meanrz.set_index('condition', inplace=True)

    clrs_post = None

    print(meanrz.columns)
    for condition, ax_condition in conditions_mapping.items():
        for objname in ['fictive_reward_shadow', 'fictive_reward']:
            arena.objects[objname]['x'] = meanrz.loc[condition, 'fictive_reward_x_px']  # mean fictive rz x
            arena.objects[objname]['y'] = meanrz.loc[condition, 'fictive_reward_y_px']  # mean actual rz x
            arena.objects[objname]['visible'] = True

        df = df_test[alltraj.condition == condition]
        testax = layout.axes[fig, 'heat_{}'.format(ax_condition)]

        clrs_post = plot_hist(testax, df.x_px, df.y_px, arena=arena, cmap=cmap, vmax=post_vmax)
        print('MAX', condition, 'test', clrs_post.get_array().max())

    # cbar_ax = layout.axes['colorbar_stim']['axis']
    f = layout.figures[fig]
    cbar_ax = layout.axes[(fig, 'colorbar')]
    cbar = f.colorbar(clrs_post, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, post_vmax])

    layout.append_figure_to_layer(f, 'mpl_layer')


def plot_ee_trajs(layout, fig, enters, exits, intersections, config):
    ax_name = lambda num: 'ax_enex{}'.format(num)  # 1-6
    print("ax 1:", ax_name(1))
    get_ax = lambda num: layout.axes[(fig, ax_name(num))]
    print(get_ax(1))
    print('loading trajectories')
    # print(intersections.head)
    print(intersections.head())

    segments = intersections.sort_values('max_intersect_len', ascending=False).set_index('fly')
    print('sorted:', segments.head())

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    ntrajs= config['ntrajs']
    scale = config['plot_scalebar']

    for i, fly in enumerate(segments.index):
        if i >= ntrajs:
            break
        flyenter = enters[enters.fly == fly]
        enterx = flyenter.x_cm - flyenter.x_cm.iloc[-1]
        entery = flyenter.y_cm - flyenter.y_cm.iloc[-1]

        flyexit = exits[exits.fly == fly]
        exitx = flyexit.x_cm - flyexit.x_cm.iloc[0]
        exity = flyexit.y_cm - flyexit.y_cm.iloc[0]

        xmin = min(enterx.min(), exitx.min())
        ymin = min(entery.min(), exity.min())
        xmax = max(enterx.max(), exitx.max())
        ymax = max(entery.max(), exity.max())
        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)

        #     my_angle = enter_exit_vectors.set_index('fly').loc[fly].angle_enter_exit
        my_seglen = segments.loc[fly, 'max_intersect_len']
        my_condition_label = segments.loc[fly, 'condition'][:-5]  # rewarded -> rew; non-rewarded -> non-rew
        ax = get_ax(i+1)

        plot_trajectory(exitx, exity, ax, markersize=config['traj_markersize'],
                        colorful=False, color=config['color_exit'])
        plot_trajectory(enterx, entery, ax, markersize=config['traj_markersize'],
                        colorful=False, color=config['color_enter'])
        ax.set_xlim(-1.15, 1.85)
        ax.set_ylim(-0.9, 2.1)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_title('{:.2f}\n{}'.format(my_seglen, my_condition_label), fontsize=7)

    if scale:
        print('plotting scale bar')
        ax = get_ax(1)
        ax.plot([-0.5, 0.5], [-0.8, -0.8], lw=2, color='k')

    print('x:',min(xmins), max(xmaxs))
    print('y:', min(ymins), max(ymaxs))


def plot_swarm_intersections(layout, fig, intersections):
    swarm_palette = ['red', 'black']
    ax_swarm = layout.axes[(fig, 'enex_all')]
    sns.swarmplot(x='condition', y='max_intersect_len', data=intersections,
                  order=['rewarded', 'non-rewarded'], palette=swarm_palette, size=3,
                  ax=ax_swarm["axis"])
    # splot = sns.boxplot(x='condition', y='max_intersect_len', data=intersections,
    #                     order=['rewarded', 'non-rewarded'],
    #                     showcaps=False, boxprops={'facecolor': 'None'},
    #                     showfliers=False, whiskerprops={'linewidth': 0}, ax=ax_swarm)
    fifi.mpl_functions.adjust_spines(ax_swarm, ['left', 'bottom'],
                                     spine_locations={'left': 0, 'bottom': 0}, spine_location_offset=0,
                                     yticks=[0, 0.75, 1.5], direction='out',
                                     smart_bounds=False, tick_length=2)
    ax_swarm.set_xticklabels(['rew', 'non-rew'])
    # for tl in ax_swarm.get_xticklabels():
    #     tl.set_visible(False)

    ax_swarm.set_ylabel('Intersection, cm')
    ax_swarm.set_xlabel('')


def plot_enter_exit(layout, fig, config):

    enters = pd.read_csv(config['enters'])
    exits = pd.read_csv(config['exits'])
    intersections = pd.read_csv(config['stats'], sep='\t')
    print(enters.shape, exits.shape)
    thr = config['threshold']
    plot_scalebar = config['plot_scalebar']
    intersections = intersections[intersections.threshold == thr]

    plot_ee_trajs(layout, fig, enters, exits, intersections, config=config)
    plot_swarm_intersections(layout, fig, intersections)

    layout.append_figure_to_layer(layout.figures[fig], 'mpl_layer')


if __name__ == '__main__':
    # layout_fname = 'layouts/layout_s2_b.svg'
    # fig_fname = 'output/figS2_0525.svg'
    layout_fname = 'fig_layouts/layout_s_analysis.svg'
    fig_fname = 'figures/S_analysis.svg'

    layout = FigureLayout(layout_fname, autogenlayers=True, make_mplfigures=True,
                          hide_layers=['fifi_axs', 'layer_trash'])
    print(layout.figures)

    alltrajs = load_trajectories(_config_['alltraj_fname'], test_period_sec=100)
    _config_['coords']['alltraj'] = alltrajs
    _config_['distance_rz']['alltraj'] = alltrajs
    _config_['walking_hists']['alltraj'] = alltrajs

    plot_downsampling(layout, 'fig_downsample', config=_config_['downsample'])
    plot_coord_transform(layout, 'fig_coord_transform', _config_['coords'])
    plot_directions_stats(layout, 'fig_directions', config=_config_['directions'])
    plot_dist_hists(layout, 'fig_dist_distrib', config=_config_['distance_rz'])
    plot_walking_hists(layout, 'fig_heat', config=_config_['walking_hists'])
    plot_enter_exit(layout, 'fig_enex', config=_config_['enter_exit'])

    layout.write_svg(fig_fname)
