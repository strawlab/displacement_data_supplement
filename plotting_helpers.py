import numpy as np
from matplotlib import cm, patches as mpatches
from matplotlib.colors import ListedColormap

conditions_mapping = {'rewarded': 'rew',
                      'non-rewarded': 'nonrew'
                      }


def my_gray_colormap(k1=0.25, k2=0.75, cmap='Greys'):
    greys_big = cm.get_cmap(cmap, 512)
    newcmp = ListedColormap(greys_big(np.linspace(k1, k2, 256)))
    return newcmp


def my_arrow(ax, x, y, dx, dy, linewidth=1, color='k', head_len=0.05, head_ratio=0.75, **kwargs):
    dx = dx * (1 - head_len)
    dy = dy * (1 - head_len)
    ar_length = np.sqrt(dx ** 2 + dy ** 2)
    head_length = head_len * ar_length
    head_width = head_length * head_ratio
    return ax.arrow(x, y, dx, dy, head_width=head_width, head_length=head_length, length_includes_head=True,
                    linewidth=linewidth, ec=color, fc=color, capstyle="butt", **kwargs)


def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5 * height, width, 0, length_includes_head=True, head_width=0.75 * height)
    return p


def plot_arc_polar(angle_from, angle_to, r, ax, **kwargs):
    arc_angles = np.linspace(angle_from, angle_to)
    rs = np.ones_like(arc_angles) * r
    ax.plot(arc_angles, rs, **kwargs)


def plot_step_hist(bins, values, ax, errorbars=None, **kwargs):
    left, right = bins[:-1], bins[1:]
    X = np.array([left, right]).T.flatten()
    Y = np.array([values, values]).T.flatten()
    ax.plot(X, Y, **kwargs)
    if errorbars is not None:
        #         err_kw = {k: v for k,v in kwargs.items() if k in ['c', 'color']}
        ecolor = kwargs['color']
        cs = (left + right) / 2
        ax.bar(cs, values, ec='none', color='none', yerr=errorbars, ecolor=ecolor)  # plot errorbars only


def mypolarhist(angles, ax, rscatter=10, scatter_size=5,
                ticklabels=True, grid=False, ax_off=True, scatter_alpha=1.0, **kwargs):
    hist_bins_angles = np.linspace(-np.pi, np.pi, 9)
    #     print('angles:',np.rad2deg(angles))
    #     shifted_angles = np.unwrap(angles - np.pi/8)
    shifted_angles = (angles - np.pi / 8 + np.pi) % (2 * np.pi) - np.pi
    #     print('shifted_angles:',np.rad2deg(shifted_angles))
    ax.scatter(x=shifted_angles, y=np.ones_like(angles) * rscatter, edgecolors='none',
               color='k', s=scatter_size, alpha=scatter_alpha)

    ax.hist(shifted_angles, bins=hist_bins_angles, edgecolor='k')
    ax.set_theta_offset(np.pi / 8 + np.pi / 2)
    #     ax.set_xticks(-np.pi/8 + np.linspace(np.pi, -np.pi, 8, endpoint=False))

    if ticklabels:
        ax.set_xticks(-np.pi / 8 + np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels(['0', '', r'- $\frac{\pi}{2}$', '', r'$\pm\pi$', '', r'$\frac{\pi}{2}$', ''])
    else:
        ax.set_xticks([])
    ax.set_thetalim(-np.pi / 8, 2 * np.pi - np.pi / 8)
    # lines, labels = ax.set_thetagrids(np.arange(-180/8., 2*180, 2*180./8))
    ax.grid(grid)
    if ax_off:
        ax.axis('off')


def plot_arcs(ax, df_teststart, align_to='angle_fr', binsadd=-np.pi / 8, r_fr=9, r_ar=9.5, alpha=0.05, lw=3):
    # for ic, condition in enumerate(['rewarded', 'non-rewarded']):

    for flyid, flydata in df_teststart.iterrows():
        fr_arc = np.array([flydata.angle_fr - flydata.fr_span, flydata.angle_fr + flydata.fr_span])
        ar_arc = np.array([flydata.angle_ar - flydata.ar_span, flydata.angle_ar + flydata.ar_span])

        # print(flydata)
        # print(flydata['angle_fr'])
        # print(flydata['anlge_reloc_start'])
        align_angle = flydata[align_to]

        show_fr_arc = fr_arc - align_angle + binsadd
        show_ar_arc = ar_arc - align_angle + binsadd

        plot_arc_polar(show_fr_arc[0], show_fr_arc[1], r_fr, ax, alpha=alpha, color='orange', lw=lw)
        plot_arc_polar(show_ar_arc[0], show_ar_arc[1], r_ar, ax, alpha=alpha, color='red', lw=lw)

