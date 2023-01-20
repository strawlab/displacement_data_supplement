import pandas as pd
import os
from figurefirst import FigureLayout
import matplotlib as mpl

from arena import plot_trajectory, load_arena_pickle
from plotting_helpers import my_gray_colormap

mpl.rc('font', size=7)

DEFAULT_ARENA_FILE = 'data/big_arena_fr_black_shadow.pickle'
ARENA = None
if os.path.isfile(DEFAULT_ARENA_FILE):
    ARENA = load_arena_pickle(DEFAULT_ARENA_FILE)

segments_mapping = {'baseline': 'baseline',
                    'stimulation': 'stim',
                    'relocation': 'reloc',
                    'test_before_movement': 'poststim',
                    'after_relocation': 'test'}


def plot_traj(ax, x, y, **kwargs):
    arena = kwargs.get('arena', ARENA)
    colorful = kwargs.get('colorful', True)
    unrew = kwargs.get('unrew', True)
    cmap = kwargs.get('cmap', 'winter')
    markersize = kwargs.get('markersize', 5)
    frew_coords = kwargs.get('fictive_reward', (arena.objects['reward']['x'], arena.objects['reward']['y']))
    # segment = segment.iloc[0]
    # foodx = foodx.iloc[0]
    # foody = foody.iloc[0]
    if unrew:
        if frew_coords[0] == arena.objects['reward']['x']:
            arena.objects['fictive_reward']['visible'] = False
        else:
            arena.objects['fictive_reward']['visible'] = True
            arena.objects['fictive_reward']['x'] = frew_coords[0]
            arena.objects['fictive_reward']['y'] = frew_coords[1]

    arena.plot(ax, axes_visible=False, margin=0.02, with_centers=False)
    ax.axis('off')
    clrs = plot_trajectory(x, y, ax, colorful=colorful, cmap=cmap, markersize=markersize, tmax=1)
    arena.objects['fictive_reward']['visible'] = False
    return clrs


def figure1(alltraj_fname, fly_ids, layout_fname, output_fname, test_period_sec=100, **kwargs):
    layout = FigureLayout(layout_fname, autogenlayers=True, make_mplfigures=True,
                          hide_layers=['fifi_axs'])

    figname = 'fig_trajs'
    # layout = FigureLayout(layout_fname)
    print(layout.figures)

    alltraj = pd.read_csv(alltraj_fname)
    print(alltraj.columns)

    clrs = None
    for i, flyid in enumerate(fly_ids):
        for segment, ax_name in segments_mapping.items():
            kw = kwargs.copy()
            ax = layout.axes[(figname, '{}{}'.format(ax_name, i+1))]
            df_fly = alltraj[(alltraj.fly == flyid) & (alltraj.segment == segment)]
            if segment == 'after_relocation':
                df_fly = df_fly[df_fly.tseg <= test_period_sec]  # plot only 100 sec after relocation as test period
                kw['fictive_reward'] = (df_fly.iloc[0].estimated_food_x, df_fly.iloc[0].estimated_food_y)  # plot fictive reward

            clrs = plot_traj(ax, df_fly.x_px, df_fly.y_px, **kw)

    cbar_ax = layout.axes[(figname, 'ax_colorbar')]
    f = layout.figures[figname]
    cbar = f.colorbar(clrs, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, 1])

    layout.append_figure_to_layer(layout.figures[figname], 'mpl_layer', cleartarget=True)
    layout.write_svg(output_fname)


if __name__ == '__main__':
    # todo: move parameters to config:
    # layout_fname,
    # flyids(3x),
    # fig_filename,
    # data_source (should contain flyid!),
    # test_period length (not implemented yet)
    # arena?

    layout_fname = 'fig_layouts/fig_displacement_layout.svg'
    fig_fname = 'figures/figure_displacement_overview.svg'

    #flyids = [6, 9, 22]
    flyids = [23, 22]  # rewarded (10 is also good), non-rewarded

    alltraj_fname = "data/all_ds_t01_cm.csv.gz"
    if not os.path.isfile(alltraj_fname):
        raise FileNotFoundError(alltraj_fname)

    figure1(alltraj_fname, flyids, layout_fname, fig_fname, cmap=my_gray_colormap(0.25, 1.0))
    # figure1(alltraj_fname, flyids, layout_fname, fig_fname)
