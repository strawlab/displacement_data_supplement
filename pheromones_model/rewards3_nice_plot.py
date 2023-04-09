import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

mpl.rc('font', size=7)

# mpl.rcParams['font.family'] = ['sans-serif']
# mpl.rcParams['font.sans-serif'] = ['Arial']
# mpl.rcParams['text.usetex'] = False
# mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['mathtext.fontset'] = 'cm'

if __name__ == '__main__':
    nbins = 25
    data_filename = sys.argv[1]
    df = pd.read_csv(data_filename[:-4] + "examples.csv")
        
    fig2, axs = plt.subplots(3, 1, figsize=(4, 4), sharex=True, sharey=True)
    iax = 0
    title_dict = {0: "Final food: middle", 1: "Final food: top", 2: "Final food: bottom"}

    tk = 2.
    for flyid, data in df.groupby('flyid'):
        axs[iax].plot(data.t/tk, data.angle, color='k')
        axs[iax].plot(data[data.eating].t/tk, data[data.eating].angle, '.', color='red', ms=3)
        axs[iax].plot(data[data.smelling].t/tk, data[data.smelling].angle, '.', color='cyan', ms=3)
        axs[iax].axhline(data.last_food_coord.iloc[0], ls='--')
        axs[iax].axhline(data.last_food_coord.iloc[0] + 2 * np.pi, ls='--')
        axs[iax].axhline(data.last_food_coord.iloc[0] - 2 * np.pi, ls='--')
        axs[iax].axvline(600/tk)
        axs[iax].set_xlim(400/tk, 1000/tk)
        axs[iax].set_title(title_dict[data.last_food_index.iloc[0]])
        axs[iax].set_ylabel('Angular postition, radians')
        iax += 1
    axs[2].set_xlabel("Time, sec")

    plt.tight_layout()
    plt.savefig(data_filename[:-4]+"_examples.svg")

    post_runs_good = pd.read_csv(data_filename[:-4] + "_postAP_runs_selected.csv")
    fig3, ax = plt.subplots(figsize=(2, 1.5))
    sns.histplot(data=post_runs_good, x='theta_midpoint', hue='last_food_index',
                 element='step', fill=False, stat='density', common_norm=False, bins=nbins, ax=ax)
    ax.set_title("Distribution of run midpoints")
    ax.set_xticks([-np.pi, 0, np.pi])
    # ax.set_xticklabels([r'- $\pi$', '0', r'$\pi$'])

    # ax.set_xticklabels(['-1', '0', '1'])
    ax.set_xticklabels(['-π', '0', 'π'])
    plt.tight_layout()
    plt.savefig(data_filename[:-4]+"_run_mids.svg")
    print("bye:)")
