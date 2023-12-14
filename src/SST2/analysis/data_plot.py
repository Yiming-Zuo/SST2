#!/usr/bin/env python3
# coding: utf-8

import os
import math

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import openmm.unit as unit
from scipy.ndimage import gaussian_filter1d


def read_ST_data(
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
    lambda_T_ref=None
):
    """
    
    Read the sst2 data from the csv files.
    The data may be splited in several files if simulation
    had to restart. The function merge all the files
    in one dataframe.

    Parameters
    ----------
    generic_name : str
        Generic name of the csv files (without the `.csv` extension).
    dt : float, optional
        Time step in ps of the simulation. The default is 0.004 ps.
    fields : list, optional
        List of the fields to read in the csv files. The default is [
        "Steps",
        "Aim Temp (K)",
        "E solute scaled (kJ/mole)",
        "E solute not scaled (kJ/mole)",
        "E solvent (kJ/mole)",
        "E solvent-solute (kJ/mole)",
        ].
    full_sep : str, optional
        Separator used in the full csv file. The default is ",".
    save_step_dcd : int, optional
        Step number used in the dcd file. The default is 100000.
    lambda_T_ref : float, optional
        Reference temperature for the lambda. The default is None.
    """

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

    # Add lambda column:
    if lambda_T_ref is not None:
        df_all["lambda"] = lambda_T_ref / df_all["Aim Temp (K)"]

        df_all[r"$\lambda$"] = pd.Series(df_all["lambda"].round(2), dtype="category")
        # inverse category order, to have higher lambda at the top
        df_all[r"$\lambda$"] = df_all[r"$\lambda$"].cat.reorder_categories(
            df_all[r"$\lambda$"].cat.categories[::-1]
        )
    # Remove Nan rows (rare cases of crashes)
    df_all = df_all[df_all["Step"].notna()]

    return df_all


def compute_exchange_prob(
    df,
    temp_col="Aim Temp (K)",
    time_ax_name=r"$Time\;(\mu s)$",
    exchange_time=2):
    """
    Compute the exchange probability and the round trip time
    for a given dataframe.
    The dataframe should be the result of a SST2 simulation.
    The dataframe should have a column with the temperature
    and a column with the time.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the SST2 simulation data.
    temp_col : str, optional
        Name of the column with the temperature. The default is "Aim Temp (K)".
    time_ax_name : str, optional
        Name of the column with the time. The default is r"$Time\;(\mu s)$".
    exchange_time : float, optional
        Time in ps between two exchange. The default is 2 ps.

    Returns
    -------
    ex_prob : float
        Exchange probability.
    trip_time : float
        Round trip time in ns.
    """

    temp_list = df[temp_col].unique()
    min_temp = temp_list[0]
    max_temp = temp_list[-1]
    print(f"Min_temp = {min_temp:.2f}, Max temp. = {max_temp:.2f}, #Rungs = {len(temp_list)}")

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
    step_time *= 1e6 # ps
    print(f"Step time = {step_time:.2f} ps")

    trip_flag = False

    all_temp_change_num = {temp: 0 for temp in temp_list}
    all_temp_num = {temp: 0 for temp in temp_list}
    temp_change_index = []
    change_index = 0
    sign = 1

    for temp in df[temp_col]:
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

    ax = sns.barplot(
        x=temp_list,
        y=vals,
        hue=temp_list)
    plt.xlabel(temp_col)
    plt.ylabel(r"$p()$")
    plt.title(r"Transition probability at each rung")
    ax.get_legend().remove()

    # print(all_temp_change_num)
    print(exchange_time / step_time)

    ex_prob = temp_change_num / len(df) * exchange_time / step_time

    if len(time_list) != 0:
        trip_time = sum(time_list) / len(time_list) / 1000  # ns
    else:
        trip_time = None

    return ex_prob, trip_time


def plot_lineplot_avg(
    df,
    x,
    y,
    quant=None,
    color="black",
    max_data=50000,
    avg_win=1000):
    """
    Plot a lineplot with a gaussian filter on the y axis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the data to plot.
    x : str
        Name of the column with the x axis data.
    y : str
        Name of the column with the y axis data.
    quant : float, optional
        Quantile to use to filter the data. The default is None.
    color : str, optional
        Color of the line. The default is "black".
    max_data : int, optional
        Maximum number of data point to plot. The default is 50000.
    avg_win : int, optional
        Window size of the gaussian filter. The default is 1000.
    
    Returns
    -------
    g : matplotlib.axes._subplots.AxesSubplot
        Axes of the plot.
    """

    local_df = filter_df(df, max_data)

    local_df["avg"] = gaussian_filter1d(local_df[y], avg_win)

    g = sns.lineplot(data=local_df, x=x, y=y, lw=0.1, color=color, alpha=0.3)
    sns.lineplot(data=local_df, x=x, y="avg", lw=2, color=color)
    if quant is not None:
        x_min, x_max = get_quant_min_max(local_df[x], quant)
        y_min, y_max = get_quant_min_max(local_df[y], quant)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    return g


def filter_df(df, max_point_number):
    """
    Filter a dataframe to keep a maximum number of data point.
    The dataframe is filtered with a step size computed
    to keep the maximum number of data point.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to filter.
    max_point_number : int
        Maximum number of data point to keep.

    Returns
    -------
    local_df : pandas.DataFrame
        Filtered dataframe.
    """

    steps_num = len(df)

    md_step = 1

    while steps_num > max_point_number:
        md_step += 1
        steps_num = len(df) // md_step

    local_df = df.loc[::md_step]
    local_df = local_df.reset_index(drop=True)

    return local_df


def get_quant_min_max(pd_serie, quant=0.001):
    """
    Get the min and max value of a pandas serie
    from a quantile.

    Parameters
    ----------
    pd_serie : pandas.Series
        Pandas serie to analyze.
    quant : float, optional
        Quantile to use. The default is 0.001.
    
    Returns
    -------
    val_min : float
        Minimum value.
    val_max : float
        Maximum value.
    """

    val_min = pd_serie.quantile(quant)
    val_max = pd_serie.quantile(1 - quant)

    return(val_min, val_max)


def plot_distri_norm(
    df,
    x,
    hue,
    x_label=None,
    max_data=50000,
    bins=100,
    element="step",
    quant=None):
    """
    Plot a distribution plot with a gaussian filter on the y axis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the data to plot.
    x : str
        Name of the column with the x axis data.
    hue : str
        Name of the column with the hue data.
    x_label : str, optional
        Label of the x axis. The default is None.
    max_data : int, optional
        Maximum number of data point to plot. The default is 20000.
    bins : int, optional
        Number of bins. The default is 100.
    element : str, optional
        Element of the plot. The default is "step".
    quant : float, optional
        Quantile to use to filter the data. The default is None.

    Returns
    -------
    ax1 : matplotlib.axes._subplots.AxesSubplot
        Axes of the plot.
    """


    local_df = filter_df(df, max_data)

    if x_label is None:
        x_label = x


    fig, ax1 = plt.subplots()

    g = sns.histplot(
        local_df, stat="density",
        kde=True,
        bins=bins, fill=False,
        x=x, common_norm=False,
        linewidth=1, alpha=0.3,
        hue=hue, element=element,
        ax=ax1)

    if quant is not None:
        x_min, x_max = get_quant_min_max(local_df[x], quant)
        plt.xlim(x_min, x_max)
    # g.axes.flat[0].xaxis.set_major_formatter(ticker.EngFormatter())
    # plt.legend(bbox_to_anchor=(1.01, 1.0))

    legend = ax1.get_legend()
    handles = legend.legendHandles

    # Remove alpha in legend
    for h in handles:
        h.set_alpha(1.0)

    if len(handles) > 15:
        ncol=2
        x_gap = 1.3
    else:
        ncol=1
        x_gap = 1.2

    sns.move_legend(g, "lower center", bbox_to_anchor=(x_gap, 0.), ncol=ncol, title_fontsize=14)

    plt.xlabel(x_label)
    return(ax1)


def plot_scatter(
    df, x, y, hue=None, x_label=None,
    y_label=None, quant=None, s=10, color=None,
    linewidth=0, label=None, legend="auto",
    alpha=None, max_data=50000):
    """
    Plot a scatter plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the data to plot.
    x : str
        Name of the column with the x axis data.
    y : str
        Name of the column with the y axis data.
    hue : str, optional
        Name of the column with the hue data. The default is None.
    x_label : str, optional
        Label of the x axis. The default is None.
    y_label : str, optional
        Label of the y axis. The default is None.
    quant : float, optional
        Quantile to use to filter the data. The default is None.
    s : int, optional
        Size of the points. The default is 10.
    color : str, optional
        Color of the points. The default is None.
    linewidth : float, optional
        Width of the points. The default is 0.
    label : str, optional
        Label of the plot. The default is None.
    legend : str, optional
        Position of the legend. The default is "auto".
    alpha : float, optional
        Transparency of the points. The default is None.
    max_data : int, optional
        Maximum number of data point to plot. The default is 50000.
        
    Returns
    -------
    g : matplotlib.axes._subplots.AxesSubplot
        Axes of the plot.
    """

    local_df = filter_df(df, max_data)

    g = sns.scatterplot(data=local_df,
                    x=x,
                    y=y,
                    s=s,
                    linewidth=linewidth,
                    legend=legend,
                    color=color,
                    alpha=alpha,
                    hue=hue)

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    if quant is not None:
        x_min, x_max = get_quant_min_max(local_df[x], quant)
        y_min, y_max = get_quant_min_max(local_df[y], quant)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    plt.legend(title=hue, bbox_to_anchor=(1.01, 1.0))

    return g


def plot_weight_RMSD(df,
        x=r"$Time\;(\mu s)$",
        hue='Temp (K)',
        ener='new_pot',
        time_ax_name=r"$Time\;(\mu s)$",
        final_weight_dict=None,
        max_data=50000,
        plot_weights=False):

    local_df = filter_df(df, max_data)
    temp_list = local_df['Aim Temp (K)'].unique()
    # gaussian_filter1d(local_df[y], avg_win)
    local_df = compute_moving_average(local_df, ener=ener)

    local_df = compute_weight_RMSD(
        local_df, final_weight_dict=final_weight_dict, ener=ener)

    if plot_weights:

        ax1 = sns.lineplot(
            data=local_df,
            x=x,
            y='avg_ener',
            hue=hue,
            lw=2)
        plt.ylabel(r"E $(KJ.mol^{-1})$")
        
        for temp in temp_list:
            avg_temp = local_df[local_df['Aim Temp (K)'] == temp][ener].mean()
            plt.axhline(avg_temp, lw=1, c="gray", linestyle=":")
        plt.legend(title=hue, bbox_to_anchor=(1.01, 1.0))
        plt.show()
    
    ax2 = sns.lineplot(
        data=local_df,
        x=time_ax_name,
        y='Weight RMSD',
        lw=2)
    plt.ylabel(r"RMSD $(KJ.mol^{-1})$")
    plt.title(r"Weights $f_i$ RMSD")

    return

def compute_moving_average(df, ener='new_pot', col_name='avg_ener'):

    temp_list = df['Aim Temp (K)'].unique()
    temp_list.sort()
    temp_list_avg = {temp:0 for temp in temp_list}
    temp_list_num = {temp:0 for temp in temp_list}
    mov_avg = []

    for temp, new_pot in zip(
            df['Aim Temp (K)'], 
            df[ener]):
        temp_list_num[temp] += 1
        temp_list_avg[temp] += (new_pot - temp_list_avg[temp]) / temp_list_num[temp]
        mov_avg.append(temp_list_avg[temp])

    df.loc[:, col_name] = mov_avg

    return df


def compute_weight_RMSD(df, final_weight_dict=None, ener='new_pot'):

    df.loc[:, 'Weight RMSD'] = 0

    if final_weight_dict is None:
        temp_final_avg = {}
    else:
        temp_final_avg = final_weight_dict

    temp_list = df['Aim Temp (K)'].unique()
    temp_list.sort()

    for temp in temp_list:

        if final_weight_dict is None:
            tmp_df = df[df['Aim Temp (K)'] == temp]
            temp_final_avg[temp] = tmp_df[ener].mean()
        
        last_avg_ener = np.nan
        weight_list = []
        for for_temp, avg_ener in zip(
            df['Aim Temp (K)'], 
            df['avg_ener']):
            if for_temp == temp:
                last_avg_ener = avg_ener
            weight_list.append(last_avg_ener)
        
        df.loc[:, f'weight {temp}'] = weight_list
        df['Weight RMSD'] += (df[f'weight {temp}'] - temp_final_avg[temp]) ** 2

    df['Weight RMSD'] = (df['Weight RMSD'] / len(temp_list)) ** 0.5

    return df


def plot_weight_RMSD(df,
        x=r"$Time\;(\mu s)$",
        hue='Temp (K)',
        ener='new_pot',
        time_ax_name=r"$Time\;(\mu s)$",
        final_weight_dict=None,
        max_data=50000,
        plot_weights=False):

    local_df = filter_df(df, max_data)
    temp_list = local_df['Aim Temp (K)'].unique()
    local_df = compute_moving_average(local_df, ener=ener)

    local_df = compute_weight_RMSD(
        local_df, final_weight_dict=final_weight_dict, ener=ener)

    if plot_weights:

        ax1 = sns.lineplot(
            data=local_df,
            x=x,
            y='avg_ener',
            hue=hue,
            lw=2)
        plt.ylabel(r"E $(KJ.mol^{-1})$")
        
        for temp in temp_list:
            avg_temp = local_df[local_df['Aim Temp (K)'] == temp][ener].mean()
            plt.axhline(avg_temp, lw=1, c="gray", linestyle=":")
        plt.legend(title=hue, bbox_to_anchor=(1.01, 1.0))
        plt.show()
    
    ax2 = sns.lineplot(
        data=local_df,
        x=time_ax_name,
        y='Weight RMSD',
        lw=2)
    plt.ylabel(r"RMSD $(KJ.mol^{-1})$")
    plt.title(r"Weights $f_i$ RMSD")

    return



def plot_free_energy(
        xall, yall, weights=None, ax=None, nbins=100, ncontours=100,
        avoid_zero_count=False, minener_zero=True, kT=2.479,
        vmin=None, vmax=None, cmap='nipy_spectral', cbar=True,
        cbar_label='free energy (kJ/mol)', cax=None, levels=None,
        cbar_orientation='vertical', norm=None, range=None,
        level_gap=None):
    """ Adapted from;
     https://github.com/markovmodel/PyEMMA/blob/devel/pyemma/plots/plots2d.py
    """
    
    z, xedge, yedge = np.histogram2d(
        xall, yall, bins=nbins, weights=weights, range=range)
    if avoid_zero_count:
        z = np.maximum(z, np.min(z[z.nonzero()]))
    x = 0.5 * (xedge[:-1] + xedge[1:])
    y = 0.5 * (yedge[:-1] + yedge[1:])
    
    pi = z.T / float(z.sum())
    free_energy = np.inf * np.ones(shape=z.shape)
    nonzero = pi.nonzero()
    zero = np.nonzero(pi == 0)
    free_energy[nonzero] = -np.log(pi[nonzero])
    #if minener_zero:
    free_energy[nonzero] -= np.min(free_energy[nonzero])
    free_energy *= kT

    # to show the highest free energy zones in the map,
    # replace infinity by a value slightly above the maximum free energy:
    free_energy[zero] = np.max(free_energy[nonzero])+1.0
    # and fix the levels for the colormap:
    if levels is None:
        levels = np.linspace(0, np.max(free_energy[nonzero])+0.5, ncontours)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    if levels is None and level_gap is not None:
        max_free = np.max(free_energy[nonzero])
        levels = int(max_free // level_gap)
        print(levels)

    mappable = ax.contourf(
        x, y, free_energy, ncontours, norm=norm,
        vmin=vmin, vmax=vmax, cmap=cmap,
        levels=levels)
    
    misc = dict(mappable=mappable)
    if cbar:
        cbar = fig.colorbar(mappable, ax=ax, orientation=cbar_orientation)
        cbar.set_label(cbar_label)
        misc.update(cbar=cbar)

    return fig, ax, misc


def compute_hdbscan_cluster(pca_df, min_cluster_size=50, min_samples=50):

    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples).fit(pca_df)
    labels = clusterer.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Number of cluster : {}, perc of non clustered points : {:.1f}%'.format(n_clusters_, 100*n_noise_/len(pca_df)))

    # count each cluster
    clust_dict = {}

    for i in range(-1, n_clusters_):
        if sum(labels == i) > 0:
            clust_dict[i] = sum(labels == i)

    # sort cluster as function of clust pop

    sorted_dict = {k: v for k, v in sorted(clust_dict.items(), key=lambda item: item[1], reverse=True)}

    # Create new cluster list
    new_label = np.copy(labels)

    select_list = []
    new_value_list = []
    cat_to_remove = []
    clust_new = 1

    for i, clust in enumerate(sorted_dict):

        if clust != -1:
            print(f'Cluster:{clust_new:3}   {sum(new_label == clust):5} | {sum(new_label == clust)/len(labels):.3f}')
            new_value_list.append(clust_new)
            clust_new += 1
        else:
            print(f'Not Clustered {sum(new_label == clust):5} | {sum(new_label == clust)/len(labels):.3f}')
            new_value_list.append(0)
            cat_to_remove = [0]
        select_list.append(new_label == clust)

    new_label = np.select(select_list, new_value_list, new_label)
    clust_serie = pd.Categorical(pd.Series(data=new_label)).remove_categories(cat_to_remove)

    return clust_serie


def compute_cluster_kmean(pca_df, max_cluster=20, random_state=0):

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 50,
        "random_state": random_state,
    }

    # A list holds the silhouette_coefficients for each k
    silhouette_coefficients = []
    kmeans_list = []

    for k in range(2, max_cluster + 1):
        print(f"{k}/{max_cluster}")
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(pca_df)
        kmeans_list.append(kmeans)     
        score = silhouette_score(pca_df, kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.plot(range(2, max_cluster + 1), silhouette_coefficients)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()

    index_min = np.argmin(silhouette_coefficients)
    print(f"{index_min+2} clusters optimal")

    clust_serie = pd.Categorical(
        pd.Series(
            data=kmeans_list[index_min].labels_) + 1
        ).remove_categories([])

    return clust_serie, kmeans_list[index_min].cluster_centers_


def plot_folding_fraction(df, col="RMSD (nm)", cutoff=0.18, label=None,
    start_time=0, time_ax_name=r"$Time\;(\mu s)$", recompute_temp_flag=True,
    ref_temp=300.0):

    df_time = df[(df[time_ax_name] > start_time)]

    fold_frac = compute_folding_fraction(
        df_time, col, cutoff)

    temp_list = df['Aim Temp (K)'].unique()
    temp_list.sort()

    if recompute_temp_flag:
        if ref_temp not in temp_list:
            ref_temp = temp_list[np.argmin(abs(temp_list-300))]
        temp_list = recompute_temp(df, ref_temp=ref_temp)

    plt.plot(temp_list, fold_frac, label=label)
    plt.scatter(temp_list, fold_frac)
    plt.xlabel('Temperature (K)')
    plt.ylabel('fraction folded')
    plt.ylim((0,1.0))


def compute_folding_fraction(df, col="RMSD (nm)", cutoff=0.18):

    temp_list = df['Aim Temp (K)'].unique()
    temp_list.sort()
    fold_frac = []

    for temp in temp_list:
        
        local_df = df[(df["Aim Temp (K)"] == temp)]
        
        num_frame = len(local_df)
        num_cutoff = sum(local_df[col] < cutoff)
        
        fold_frac.append(num_cutoff/num_frame)


    
    return(fold_frac)


def recompute_temp(df, ref_temp=300.0):

    # Compute real temp using Stinermann et al. JCTC 2015
    
    # Bi' = Bi(1 + ((Bref/Bi)**0.5 -1 ) ( Epw/(Epp+Epw)) )
    temp_list_traj = df['Aim Temp (K)'].unique()
    temp_list_traj.sort()

    new_temp_list = []
    ref_temp_index = list(temp_list_traj).index(ref_temp)
    
    inverseTemperatures = [1.0/(unit.MOLAR_GAS_CONSTANT_R*t) for t in temp_list_traj]
    
    for i, temp in enumerate(temp_list_traj):
        local_df = df[df["Aim Temp (K)"] == temp]
        #Epw
        Epp = local_df["E solute scaled (kJ/mole)"].mean()
        Epw = local_df["E solvent-solute (kJ/mole)"].mean()
        
        new_Bi = inverseTemperatures[i]*(1 + ((inverseTemperatures[ref_temp_index]/inverseTemperatures[i])**0.5 -1 ) * ( Epw/(Epp+Epw)) )
        new_temp = 1.0/(unit.MOLAR_GAS_CONSTANT_R*new_Bi)
        # print(i, temp, new_temp)
        new_temp_list.append(new_temp)

    return(new_temp_list)


def compute_folding_fraction_RMSD(df, col="RMSD (nm)", cutoff=0.18,
    start_time=0, time_ax_name=r"$Time\;(\mu s)$",
    ref_fold_frac = None,
    time_interval=2.0):

    if ref_fold_frac is None:
        df_time = df[(df[time_ax_name] > start_time)]
        ref_fold_frac = compute_folding_fraction(
            df_time, col, cutoff)
        ref_fold_frac = np.array(ref_fold_frac)

    temp_list = df['Aim Temp (K)'].unique()
    temp_list.sort()

    max_time = df[time_ax_name].max()

    RMSD_list = []
    time_list = []
    for i in range( int((max_time-start_time)/time_interval) + 1):
        #print(f"{i}  {start_time:.1f}  {start_time + (i + 1) * time_interval:.1f}")
    

        df_time = df[(df[time_ax_name] > start_time) &
                     (df[time_ax_name] < start_time + (i + 1) * time_interval)]
        fold_frac = []

        fold_frac = compute_folding_fraction(
            df_time, col, cutoff)
        
        fold_frac = np.array(fold_frac)
        
        RMSD_list.append(np.sum( (ref_fold_frac-fold_frac)**2))
        time_list.append(start_time + (i + 1) * time_interval)


    return time_list, RMSD_list

def plot_folding_fraction_RMSD(df, col="RMSD (nm)", cutoff=0.18, label=None,
    start_time=0, time_ax_name=r"$Time\;(\mu s)$",
    ref_fold_frac = None,
    color=None, ls='-',
    s=20, alpha=1.0,
    time_interval=2.0):

    time_list, RMSD_list = compute_folding_fraction_RMSD(
        df, col, cutoff,
        start_time, time_ax_name,
        ref_fold_frac,
        time_interval)

    if color is None:
        plt.plot(time_list, RMSD_list, label=label, ls=ls, alpha=alpha)
        plt.scatter(time_list, RMSD_list, ls=ls, s=s, alpha=alpha)
    else:
        plt.plot(time_list, RMSD_list, label=label, color=color, ls=ls, alpha=alpha)
        plt.scatter(time_list, RMSD_list, color=color, ls=ls, s=s, alpha=alpha)        
    
    plt.xlabel(time_ax_name)
    plt.ylabel('fraction folded RMSD')
    plt.legend()

    return time_list, RMSD_list


def plot_folding_fraction_convergence(df, col="RMSD (nm)", cutoff=0.18, label=None,
    start_time=0, time_ax_name=r"$Time\;(\mu s)$", recompute_temp_flag=False,
    ref_temp=300.0, time_interval=2.0):

    temp_list = df['Aim Temp (K)'].unique()
    temp_list.sort()


    if recompute_temp_flag:
        if ref_temp not in temp_list:
            ref_temp = temp_list[np.argmin(abs(temp_list-300))]
        temp_list_plot = recompute_temp(df, ref_temp=ref_temp)
    else:
        temp_list_plot = temp_list
    
    max_time = df[time_ax_name].max()
    print(max_time)
    
    for i in range( int((max_time-start_time)/time_interval) + 1):
        #print(f"{i}  {start_time:.1f}  {start_time + (i + 1) * time_interval:.1f}")
    

        df_time = df[(df[time_ax_name] > start_time) &
                     (df[time_ax_name] < start_time + (i + 1) * time_interval)]
        
        fold_frac = compute_folding_fraction(
            df_time, col, cutoff)
                
        plt.plot(temp_list_plot, fold_frac, label=f"{start_time}-{start_time + (i + 1) * time_interval:.1f}")
        plt.scatter(temp_list_plot, fold_frac)
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('fraction folded')
    plt.ylim((0,1.0))
    plt.legend()
