import sys
import time

import matplotlib
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

from shared_funcs import angle_minuspitopi


def mark_stages(data):
    data["stage"] = 'AP'
    last_eating_time = data[data.eating].iloc[-1].t
    data.loc[data.t > last_eating_time, "stage"] = 'post'
    return data


def mark_return(idata):
    idata['return_status'] = 'post_return'
    if idata[idata.relative_angle.abs() >= 2 * np.pi].empty:
        return idata
    departure_time = idata[idata.relative_angle.abs() >= 2 * np.pi].iloc[0].t

    idata.loc[idata.t < departure_time, "return_status"] = 'pre_return'
    return idata


if __name__ == '__main__':
    to_plot = True
    to_save = True
    # parameters
    AP_len = 5 * 60 * 2  # (5 minutes, 2 steps per sec)
    postAP_len = 5 * 60 * 2
    iter_len = AP_len + postAP_len
    t_start = 5  # ap starts from this time, ignore everything before.
    n_iters = 6  # number of iterations for every fly
    example_flyid = 42  # plot trajectories of one individual simulation

    data_filename = sys.argv[1]
    pdf_fname = data_filename[:-4] + ".pdf"  # csv -> pdf , in the same folder
    # data_path = os.path.join("data/circling", data_filename)
    df = pd.read_csv(data_filename)
    ######################################################################
    # preprocessing
    print("preprocessing...")

    df["iteration"] = np.floor((df.t - t_start) / iter_len)
    df = df[(df.iteration >= 0) & (df.iteration < n_iters)]

    # tt=0 when fly received reward last time, time scale for every trial

    # does not work properly??!!
    # df["tt"] = df.groupby(['flyid', 'iteration']).apply(lambda df: df.t - df[df.eating].iloc[-1].t).values
    # df['stage'] = 'AP'
    # df.loc[df.tt >= 0, 'stage'] = 'post'

    # changed 11.12 copied from notebook : mark stages, removed departure..
    df = df.groupby(['flyid', 'iteration']).apply(mark_stages)

    # analyse only postAP
    df = df[df.stage == 'post']
    # angle relative to the last reward position
    df["relative_angle"] = df.groupby(['flyid', 'iteration']).angle.apply(lambda alpha: alpha - alpha.iloc[0])
    # redundant? tt is the same?
    df["t_post"] = df.groupby(['flyid', 'iteration']).t.apply(lambda t: t - t.iloc[0])

    ######################################################################
    # get the runs
    df = df.groupby(['flyid', 'iteration']).apply(mark_return)
    pd_runs = df[df.return_status == 'post_return'].groupby(["flyid", "iteration", "run_num"]).aggregate(
        {'relative_angle': ['first', 'last'], 'direction': 'first'}).reset_index()
    pd_runs.columns = ["_".join(x) for x in pd_runs.columns.ravel()]
    pd_runs.rename(columns={"run_num_": "run_num", "flyid_": "flyid", "iteration_": "iteration",
                            "direction_first": "direction"}, inplace=True)
    pd_runs['run_midpoint'] = (pd_runs.relative_angle_last + pd_runs.relative_angle_first) / 2
    pd_runs['theta_midpoint'] = pd_runs.run_midpoint.apply(lambda angle: angle_minuspitopi(angle))
    if to_save:
        df.to_csv(data_filename[:-4] + "_postAP_preprocessed.csv", index=False)
        pd_runs.to_csv(data_filename[:-4] + "_post_return_runs.csv", index=False)

    pd_runs.to_csv('post_return_runs.csv', index=False)

    ######################################################################
    if not to_plot:
        exit(0)

    # create pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_fname)

    ######################################################################
    #  fig 1: trajectories in postAP
    print('Fig1...')
    fig1, ax = plt.subplots(figsize=(8, 4))
    one_example = df[df.flyid == example_flyid].copy()
    one_example = one_example.groupby('iteration').apply(mark_return)
    one_example.to_csv(data_filename[:-4] + '_example.csv', index=False)

    pos_scale = 2 * np.pi  # position in full revolutions
    t_scale = 2.  # time in seconds
    for i, data in one_example.groupby('iteration'):
        ax.plot(data.t_post / t_scale, data.relative_angle / pos_scale)
        ax.plot(data[data.return_status == 'pre_return'].t_post / t_scale,
                data[data.return_status == 'pre_return'].relative_angle / pos_scale, color='k')
        ax.plot(data[data.eating].t_post / t_scale, data[data.eating].relative_angle / pos_scale, '.', color='red')
        ax.plot(data[data.smelling].t_post / t_scale, data[data.smelling].relative_angle / pos_scale, '.', color='cyan')
    ax.set_xlabel('time, sec')
    ax.set_ylabel('position, revolutions')
    yticks = np.arange(-5, 6)
    ax.set_yticks(yticks)
    ax.grid(axis='y', color='0.95')
    ax.set_title("Trajectories in postAP for one simulation")
    plt.tight_layout()
    pdf.savefig(fig1)

    # fig 2: run midpoints histogram
    print('Fig2...')
    fig2, ax_hist = plt.subplots(figsize=(10, 4))
    h = ax_hist.hist(pd_runs.run_midpoint / pos_scale, bins=150)
    for i in range(5):
        ax_hist.axvline(i, c='gray', ls='--')
        ax_hist.axvline(-i, c='gray', ls='--')
    ax_hist.set_title('Run midpoints distribution')
    ax_hist.set_xlabel('Run midpoint, revolutions')

    ax_hist.set_ylabel('Count')
    plt.tight_layout()
    pdf.savefig(fig2)

    pdf.close()
    time.sleep(2)
    print("done :)")
