import sys
import time

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# def angle_minuspitopi(angle):
#     return (angle + np.pi) % (2 * np.pi) - np.pi
from shared_funcs import angle_in_range, angle_minuspitopi

if __name__ == '__main__':
    data_filename = sys.argv[1]
    df = pd.read_csv(data_filename)

    run_info = df.groupby(["flyid", "run_num"]).aggregate(
        {'angle': ['first', 'last'], 'last_food_index': 'first', 'direction': 'first'}).reset_index()
    run_info.columns = ["_".join(x) for x in run_info.columns.ravel()]
    run_info['run_midpoint'] = (run_info.angle_last + run_info.angle_first) / 2
    run_info.rename(columns={"run_num_": "run_num", "flyid_": "flyid",
                             "last_food_index_first": "last_food_index",
                             "direction_first": "direction"}, inplace=True)
    run_info['theta_midpoint'] = run_info.run_midpoint.apply(lambda angle: angle_minuspitopi(angle))
    pdf_fname = data_filename[:-4]+".pdf"
    post_runs = run_info[run_info['run_num'] >= 0]

    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_fname)

    # fig1 = plt.figure(figsize=(10, 5))
    # sns.histplot(data=post_runs, x='theta_midpoint', hue='last_food_index', element='step', fill=False, bins=20)
    # plt.title("Run length midpoints Post, all trials")
    # pdf.savefig(fig1)

    food_coords = df.last_food_coord.unique()
    for ifood, food_coord in enumerate(food_coords):
        col = f"contains_{ifood}"
        run_info[col] = run_info.apply(lambda row: angle_in_range(food_coord, row.angle_first, row.angle_last), axis=1)
    run_info["contains_food"] = run_info.contains_0.astype(int) + run_info.contains_1.astype(int) + run_info.contains_2.astype(int)

    run_info.to_csv(data_filename[:-4] + "_runs.csv", index=False)

    ap_runs = run_info[run_info.run_num < 0]
    good_ones2 = ap_runs[(ap_runs.run_num == -2) & (ap_runs.contains_food == 3)]
    good_ones3 = ap_runs[(ap_runs.run_num == -3) & (ap_runs.contains_food == 3)]
    good_flies2 = set(good_ones2.flyid.unique())
    good_flies3 = set(good_ones3.flyid.unique())
    good_flies = good_flies2.intersection(good_flies3)

    print(f"n good flies: {len(good_flies)}")
    good_ones = ap_runs[ap_runs.flyid.isin(good_flies)]

    print("numbers of good simulations: ", good_ones.last_food_index.value_counts())
    example_flyids = good_ones.groupby("last_food_index").flyid.first()

    df_examples = df[df.flyid.isin(example_flyids)]
    df_examples.to_csv(data_filename[:-4] + "_examples.csv")

    fig2, axs = plt.subplots(3, 1, figsize=(7, 8))
    for i, fly in enumerate(example_flyids):
        data = df[df.flyid == fly]
        axs[i].plot(data.t, data.angle, color='k')
        axs[i].plot(data[data.eating].t, data[data.eating].angle, '.', color='red')
        axs[i].plot(data[data.smelling].t, data[data.smelling].angle, '.', color='cyan')
        axs[i].axhline(data.last_food_coord.iloc[0], ls='--')
        axs[i].axhline(data.last_food_coord.iloc[0] + 2 * np.pi, ls='--')
        axs[i].axhline(data.last_food_coord.iloc[0] - 2 * np.pi, ls='--')
        axs[i].axvline(600)
        axs[i].set_xlim(400, 1000)
        axs[i].set_title(str(fly))
    plt.tight_layout()
    pdf.savefig(fig2)

    post_runs_good = post_runs[post_runs.flyid.isin(good_flies)]
    post_runs_good.to_csv(data_filename[:-4] + "_postAP_runs_selected.csv", index=False)

    fig3, axs = plt.subplots(2, 1, figsize=(4,6))
    # bins was 20 before
    sns.histplot(data=post_runs, x='theta_midpoint', hue='last_food_index', element='step',
                 fill=False, bins=21, ax=axs[0])
    axs[0].set_title("Run length midpoints Post, all trials")

    sns.histplot(data=post_runs[post_runs.flyid.isin(good_flies)], x='theta_midpoint', hue='last_food_index',
                 element='step', fill=False, stat='density', common_norm=False, bins=21, ax=axs[1])
    axs[1].set_title("Selected trials")
    plt.tight_layout()
    pdf.savefig(fig3)
    pdf.close()
    time.sleep(2)
    print("bye:)")
