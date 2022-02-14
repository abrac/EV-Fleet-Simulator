#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import data_processing_ev as dpr


def plot_stop_duration_boxes(scenario_dir: Path, plot_blotches: bool = False,
                             figsize=(4, 3), **kwargs):
    # For each EV, calculate the net stop_duration at each date.
    # ----------------------------------------------------------

    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])

    # Load list of stop-arrivals and -durations as DataFrame.
    stops_file = scenario_dir.joinpath('Temporal_Clusters', 'Clustered_Data',
                                       'time_clusters_with_dates.csv')

    if input_data_fmt == dpr.DATA_FMTS['GPS']:
        stops_df = pd.read_csv(stops_file).set_index(['EV_Name', 'Date'])
            # TODO convert dates to `datetime.date`s.
        # Split stops_df into one df per EV.
        ev_stop_dfs = [df for _, df in stops_df.groupby(level=[0])]
        ev_names = [ev_name for ev_name, _ in stops_df.groupby(level=[0])]
    elif input_data_fmt == dpr.DATA_FMTS['GTFS']:
        ev_stop_dfs = [pd.read_csv(stops_file).set_index(['EV_Name', 'Date'])]
        ev_names = ['Fleet']
    else:
        raise ValueError(dpr.DATA_FMT_ERROR_MSG)

    # Filter out stop events about a certain time threshold.
    filtered_ev_stop_dfs = []
    for ev_stop_df in ev_stop_dfs:
        filtered_ev_stop_df = ev_stop_df[ev_stop_df['Stop_Duration'] <= 8]
        filtered_ev_stop_dfs.append(filtered_ev_stop_df)

    # Group each stop_df by date, and calculate the sum at each date.
    summed_ev_stop_dfs = []
    for ev_stop_df in filtered_ev_stop_dfs:
        summed_ev_stop_df = ev_stop_df['Stop_Duration'].groupby(
            ['EV_Name', 'Date']).sum()
        summed_ev_stop_dfs.append(summed_ev_stop_df)

    # Plot the ev_stop_dfs as box-plots.
    plt.figure(figsize=figsize)
    plt.boxplot(summed_ev_stop_dfs,
                medianprops={'color': 'black'},
                flierprops={'marker': '.'})
    # plt.title("Daily Stop Duration per EV")
    plt.ylabel("Daily duration of stop-events (Hours)", fontsize='small')
    plt.xticks(range(1, len(ev_names) + 1), ev_names, rotation=30,
               fontsize='small')
    plt.xlabel("eMBT ID")
    ax = plt.gca()
    ax.set_ylim(ymin=0, ymax=24)

    if plot_blotches:
        # Plot the stop_events which make up the box-plots.
        for i, ev_stop_df in enumerate(summed_ev_stop_dfs):
            # Generate random x-values centered around the box-plot.
            x = np.random.normal(loc=1 + i, scale=0.04, size=len(ev_stop_df))
            plt.scatter(x, ev_stop_df, alpha=0.4)
    plt.tight_layout()

    # Save the box-plots and dataframes.
    save_dir = scenario_dir.joinpath('Temporal_Clusters', 'Clustered_Data',
                                     'stop_duration_box_plots')
    save_dir.mkdir(parents=True, exist_ok=True)
    # Output box-plots.
    # As png:
    fig_dir = save_dir.joinpath("stop_duration_box_plots.png")
    plt.savefig(fig_dir)
    # As svg:
    fig_dir = save_dir.joinpath("stop_duration_box_plots.pdf")
    plt.savefig(fig_dir)
    # As pickle:
    fig_dir = save_dir.joinpath(
        "stop_duration_box_plots.fig.pickle")
    fig = plt.gcf()
    pickle.dump(fig, open(fig_dir, 'wb'))

    ev_stop_dfs_combined = pd.concat(summed_ev_stop_dfs)
    ev_stop_dfs_combined.to_csv(
        save_dir.joinpath('stop_duration_box_plots.csv')
    )

    plt.show()


if __name__ == "__main__":
    scenario_dir = Path(os.path.abspath(__file__)).parents[2]
    _ = input('Do you want to plot "blotches" on box-plots? y/[N] ')
    blotches = True if _.lower() == 'y' else False
    plot_stop_duration_boxes(scenario_dir, blotches)
