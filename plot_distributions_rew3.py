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


# def mark_departure(idata):
#     idata['departure_status'] = 'post_departure'
#     if idata[idata.relative_angle.abs() >= np.pi].empty:
#         return idata
#     departure_time = idata[idata.relative_angle.abs() >= np.pi].iloc[0].t
#
#     idata.loc[idata.t < departure_time, "departure_status"] = 'pre_departure'
#     return idata

def mark_stages(data):
    data["stage"] = 'AP'
    last_eating_time = data[data.eating].iloc[-1].t
    data.loc[data.t > last_eating_time, "stage"] = 'post'
    angle_end = data[data.stage == 'post'].iloc[0].angle

    nrevs_f = angle_end / (2 * np.pi)
    nrevs_i = np.rint(nrevs_f)
    angle_anchor = nrevs_i * 2 * np.pi
    data["relative_angle"] = np.NaN
    data.loc[data.t > last_eating_time, "relative_angle"] = data[data.t > last_eating_time].angle - angle_anchor

    data["departure_state"] = 'pre'
    if data[data.relative_angle.abs() > np.pi].empty:
        return data
    departure_time = data[data.relative_angle.abs() > np.pi].iloc[0].t
    #     print(departure_time)
    data.loc[data.t >= departure_time, "departure_state"] = "post"

    return data


if __name__ == '__main__':
    data_filename = sys.argv[1]
    df = pd.read_csv(data_filename)
    df = df.groupby('flyid').apply(mark_stages)  # departure_state and stage

    run_info = df.groupby(["flyid", "run_num"]).aggregate(
        {'angle': ['first', 'last'],
         'last_food_index': 'first',
         'direction': 'first',
         'departure_state': 'last'}).reset_index()
    run_info.columns = ["_".join(x) for x in run_info.columns.ravel()]
    run_info['run_midpoint'] = (run_info.angle_last + run_info.angle_first) / 2
    run_info.rename(columns={"run_num_": "run_num", "flyid_": "flyid",
                             "last_food_index_first": "last_food_index",
                             "direction_first": "direction",
                             "departure_state_last": "departure_state"}, inplace=True)
    run_info['theta_midpoint'] = run_info.run_midpoint.apply(lambda angle: angle_minuspitopi(angle))
    pdf_fname = data_filename[:-4]+".pdf"
    post_runs = run_info[run_info['run_num'] >= 0]
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_fname)

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

    fig3, axs = plt.subplots(3, 1, figsize=(4, 9))
    # bins was 20 before
    bins = np.linspace(-np.pi, np.pi, 22)
    sns.histplot(data=post_runs, x='theta_midpoint', hue='last_food_index', element='step',
                 fill=False, bins=bins, ax=axs[0])
    axs[0].set_title("Run length midpoints Post, all trials")

    sns.histplot(data=post_runs[post_runs.flyid.isin(good_flies)], x='theta_midpoint', hue='last_food_index',
                 element='step', fill=False, stat='density', common_norm=False, bins=bins, ax=axs[1])
    axs[1].set_title("Selected trials")

    print(post_runs.departure_state.value_counts())
    sns.histplot(data=post_runs[(post_runs.flyid.isin(good_flies)) & (post_runs.departure_state == 'pre')],
                 x='theta_midpoint', hue='last_food_index',
                 element='step', fill=False, stat='density', common_norm=False, bins=bins, ax=axs[2])
    axs[2].set_title("Selected trials pre departure")
    axs[2].set_xticks([-np.pi, 0, np.pi])
    axs[2].set_xticklabels([r'- $\pi$', '0', r'$\pi$'])

    plt.tight_layout()
    pdf.savefig(fig3)

    pdf.close()
    time.sleep(2)
    print("bye:)")
