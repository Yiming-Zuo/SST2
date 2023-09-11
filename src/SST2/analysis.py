import os
import math

import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d


def read_sst2_data(
    generic_name,
    dt=0.004,
    fields=[
        "Steps",
        "Aim Temp (K)",
        "E solute scaled (kJ/mole)",
        "E solute not scaled (kJ/mole)",
        "E solvent (kJ/mole)",
        "E solvent-solute (kJ/mole)",
    ],
    full_sep=",",
    save_step_dcd=100000,
):
    # Get part number
    part = 1
    while os.path.isfile(f"{generic_name}_part_{part + 1}.csv"):
        part += 1
    print(f" part number = {part}")

    df_temp = pd.read_csv(f"{generic_name}_full.csv", usecols=fields, sep=full_sep)
    df_sim = pd.read_csv(f"{generic_name}.csv")


    for i in range(2, part + 1):

        last_old_step = df_temp.iloc[df_temp.index[-1], 0]

        print(f"Reading part {i}")
        df_sim_part = pd.read_csv(f"{generic_name}_part_{i}.csv")
        df_temp_part = pd.read_csv(
            f"{generic_name}_full_part_{i}.csv", usecols=fields, sep=full_sep
        )

        # print(df_temp_part.head(1))

        # Read step
        first_new_step = df_temp_part.iloc[0, 0]

        print(first_new_step)

        # The dcd format has some limitation in the number of step
        # In some case a simulation restart has to define step number at 0
        # To avoid issues with the dcd. The last step has to be actualize
        # as function of the last simulation.
        if first_new_step < last_old_step - save_step_dcd:
            chk_step = (
                df_sim['#"Step"'][df_sim['#"Step"'] % save_step_dcd == 0].iloc[-1]
                - first_new_step
            )
            df_temp_part[fields[0]] += chk_step
            df_sim_part['#"Step"'] += chk_step
            print(f"add {chk_step} to {generic_name}_full_part_{i}.csv")

        df_sim = (
            pd.concat([df_sim, df_sim_part], axis=0, join="outer")
            .reset_index()
            .drop(["index"], axis=1)
        )
        df_temp = (
            pd.concat([df_temp, df_temp_part], axis=0, join="outer")
            .reset_index()
            .drop(["index"], axis=1)
        )
        del df_sim_part, df_temp_part

    print(f"sim csv  : {len(df_sim)}")
    print(f"temp csv : {len(df_temp)}")

    if '#"Steps"' in df_temp.columns:
        df_temp = df_temp.rename(
            columns={'#"Steps"': "Step", "Temperature (K)": "Aim Temp (K)"}
        )

    max_step = min([len(df_sim), len(df_temp)])
    print(max_step)

    df_sim = df_sim.iloc[:max_step]
    df_temp = df_temp.iloc[:max_step]
    # Add time column
    df_temp[r"$Time\;(\mu s)$"] = df_temp["Step"] * dt / 1e6
    df_sim = df_sim.drop(['#"Step"'], axis=1)

    # Concat both dataframe
    df_all = pd.concat([df_temp, df_sim], axis=1)
    del df_temp, df_sim

    # Add a categorical column for temp
    df_all["Temp (K)"] = pd.Categorical(df_all["Aim Temp (K)"])

    # Remove Nan rows (rare cases of crashes)
    df_all = df_all[df_all["Step"].notna()]

    return df_all


def compute_exchange_prob(df, time_ax_name=r"$Time\;(\mu s)$", exchange_time=2):
    temp_list = df["Aim Temp (K)"].unique()
    min_temp = temp_list[0]
    max_temp = temp_list[-1]

    # Compute exchange probability:
    last_temp = min_temp
    temp_change_num = 0

    # Compute Round trip time
    # or time to go from min to
    # max and back to min temp:

    time_list = []
    target_temp = last_temp
    step_num = 0

    step_time = df.loc[1, time_ax_name] - df.loc[0, time_ax_name]
    step_time *= 1e6

    trip_flag = False

    temp_list = df["Aim Temp (K)"].unique()
    temp_list.sort()
    all_temp_change_num = {temp: 0 for temp in temp_list}
    all_temp_num = {temp: 0 for temp in temp_list}
    temp_change_index = []
    change_index = 0
    sign = 1

    for temp in df["Aim Temp (K)"]:
        all_temp_num[temp] += 1
        # round trip time
        if temp == min_temp:
            if trip_flag:
                time_list.append(step_num * step_time)
            step_num = 0
            trip_flag = False
        elif temp == max_temp:
            step_num += 1
            trip_flag = True
        else:
            step_num += 1
        # Exchange prob
        if temp != last_temp:
            all_temp_change_num[temp] += 1
            temp_change_num += 1
            sign = temp - last_temp
            last_temp = temp
            change_index = 1
        else:
            change_index += 1
        temp_change_index.append(math.copysign(change_index, sign))

    # print(temp_change_index)
    df["Temp Change index"] = temp_change_index

    for temp in temp_list:
        all_temp_change_num[temp] /= all_temp_num[temp]
        all_temp_change_num[temp] *= exchange_time / step_time

    keys = list(all_temp_change_num.keys())
    # get values in the same order as keys, and parse percentage values
    vals = [all_temp_change_num[k] for k in keys]
    sns.barplot(x=list(range(len(keys))), y=vals)
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$p()$")
    plt.title(r"Transition probability for each $\lambda$ rung")

    # print(all_temp_change_num)
    print(exchange_time / step_time)

    ex_prob = temp_change_num / len(df) * exchange_time / step_time

    if len(time_list) != 0:
        trip_time = sum(time_list) / len(time_list) / 1000  # ns
    else:
        trip_time = None

    return ex_prob, trip_time


def plot_lineplot_avg(df, x, y, quant=None, max_data=50000, avg_win=1000):
    local_df = filter_df(df, max_data)

    if quant is not None:
        quant = 0.001
        x_min = local_df[x].quantile(quant)
        x_max = local_df[x].quantile(1 - quant)
        y_min = local_df[y].quantile(quant)
        y_max = local_df[y].quantile(1 - quant)

    local_df["avg"] = gaussian_filter1d(local_df[y], avg_win)

    g = sns.lineplot(data=local_df, x=x, y=y, lw=0.1)
    sns.lineplot(data=local_df, x=x, y="avg", lw=2)
    if quant is not None:
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    return g


def filter_df(df, max_point_number):
    steps_num = len(df)

    md_step = 1

    while steps_num > max_point_number:
        md_step += 1
        steps_num = len(df) // md_step

    local_df = df.loc[::md_step]
    local_df = local_df.reset_index(drop=True)

    return local_df
