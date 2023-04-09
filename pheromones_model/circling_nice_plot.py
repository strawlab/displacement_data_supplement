import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# https://matplotlib.org/stable/tutorials/text/mathtext.html did not work
mpl.rc('font', size=6)
plt.rcParams['svg.fonttype'] = 'none'

if __name__ == '__main__':
    example_flyid = 42
    nbins = 111
    rawdata_filename = sys.argv[1]
    svg_fname = rawdata_filename[:-4] + ".svg"

    df = pd.read_csv(rawdata_filename[:-4] + "_postAP_preprocessed.csv")
    pd_runs = pd.read_csv(rawdata_filename[:-4] + "_post_return_runs.csv")

    f, (ax_ex, ax_hist) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 2]}, sharey=True, figsize=(6, 3))

    one_example = df[df.flyid == example_flyid].copy()
    pos_scale = 2 * np.pi  # position in full revolutions
    t_scale = 2.  # time in seconds
    for i, data in one_example.groupby('iteration'):
        ax_ex.plot(data.t_post / t_scale, data.relative_angle / pos_scale, lw=0.7)
        ax_ex.plot(data[data.return_status == 'pre_return'].t_post / t_scale,
                   data[data.return_status == 'pre_return'].relative_angle / pos_scale, color='k', lw=0.7)
        ax_ex.plot(data[data.eating].t_post / t_scale, data[data.eating].relative_angle / pos_scale, '.', color='red')
        ax_ex.plot(data[data.smelling].t_post / t_scale, data[data.smelling].relative_angle / pos_scale, '.',
                   color='cyan', ms=3)
    ax_ex.set_xlabel('Time, sec')
    ax_ex.set_ylabel('Position, revolutions')
    yticks = np.arange(-5, 6)
    ax_ex.set_yticks(yticks)
    ax_ex.grid(axis='y', color='0.95')
    ax_ex.set_title("Trajectories in postAP for one simulation")

    h = ax_hist.hist(pd_runs.run_midpoint / pos_scale, bins=nbins,
                     orientation='horizontal', zorder=3, histtype='step', color='k')
    ax_hist.grid(axis='y', color='0.95', zorder=0)
    ax_hist.set_title('Run midpoints distribution')
    #ax_hist.xlabel('Run midpoint, revolutions')
    ax_hist.set_xlabel('Count')
    plt.ylim(-5, 5)
    plt.tight_layout()
    plt.savefig(svg_fname)
