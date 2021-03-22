#! /usr/bin/env python3
"""
This scripts generates box plots from the stats.json files. The script must be
executed from the within the scenario's simulation-outputs directory.
"""
# FIXME Incorporate this into "Results Analysis" module of code repository.

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import repeat
import pickle

if __name__ == "__main__":
    plot_blotches = False
    figsize = (4, 3)
    # Get a list of paths to the stats.json files of each taxi.
    simulation_outputs_dir = Path(__file__).parent  # TODO Make this relative
        # to scenario_dir.
    ev_stats_files = sorted([
        *simulation_outputs_dir.glob('T*/Outputs/stats_*.json')
    ])

    # for each ev, read statistics and generate a box plot
    all_date_strs = []
    all_energy_diffs = []
    ev_names = []
    plt.figure(figsize=figsize)
    for i, ev_stats_file in enumerate(ev_stats_files):
        with open(ev_stats_file, 'r') as f:
            ev_stats = json.load(f)

        # Get a list of energy_diffs which are present in ev_stats.
        dates = ev_stats['stats']['dates']
        energy_diffs = [-date["energy diffs"]["00:00:00 -> 23:59:59"] for
                        date in dates]
        all_energy_diffs.append(energy_diffs)

        # Get a list of dates (as strings) which are present in ev_stats.
        date_strs = [date['date'] for date in dates]
        all_date_strs.append(date_strs)

        if plot_blotches:
            # Generate random x-values centered around the box-plot.
            x = np.random.normal(loc=1+i, scale=0.04, size=len(energy_diffs))
            plt.scatter(x, energy_diffs, alpha=0.4)

        ev_names.append(ev_stats_file.stem.split('_')[1])

    # Generate a box-plot from the values in all_energy_diffs.
    plt.boxplot(all_energy_diffs,
                medianprops={'color': 'black'},
                flierprops={'marker': '.'})
    #plt.title("Daily energy usage per taxi")
    plt.ylabel("Daily energy usage (kWh)")
    plt.xticks(range(1, len(ev_names)+1), ev_names, rotation=30,
               fontsize='small')
    plt.xlabel("eMBT ID")
    plt.tight_layout()

    # Output box-plots.
    # As png:
    fig_dir = simulation_outputs_dir.joinpath("Energy_usage_box_plots.png")
    plt.savefig(fig_dir)
    # As svg:
    fig_dir = simulation_outputs_dir.joinpath("Energy_usage_box_plots.pdf")
    plt.savefig(fig_dir)
    # As pickle:
    fig_dir = simulation_outputs_dir.joinpath(
        "Energy_usage_box_plots.fig.pickle")
    fig = plt.gcf()
    pickle.dump(fig, open(fig_dir, 'wb'))

    # Convert all_energy_diffs to a dataframe and save as a csv file.
    csv_dir = simulation_outputs_dir.joinpath("Energy_usage.csv")
        # TODO Make this relative to scenario_dir.
    export_data_zipped = []
    for i, ev_name in enumerate(ev_names):
        ev_name_list = [*repeat(ev_name, len(all_date_strs[i]))]
        export_data_zipped.extend(
            [*zip(ev_name_list, all_date_strs[i], all_energy_diffs[i])]
        )

    df = pd.DataFrame(export_data_zipped,
                      columns=['ev_name', 'date', 'energy_used'])
    df.set_index(['ev_name', 'date'])  # This is not necessary for the saving
        # process, but it's useful in-case this script is extended.
    df.to_csv(csv_dir, index=False)

    plt.show()
