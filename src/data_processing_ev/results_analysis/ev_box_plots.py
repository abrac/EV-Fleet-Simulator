#! /usr/bin/env python3
"""
This scripts generates box plots from the stats.json files. The script must be
executed from the within the scenario's simulation-outputs directory.
"""

# TODO: GTFS implementation of ev_box_plots should consider frequencies.txt.

from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from itertools import repeat
import pickle
import data_processing_ev as dpr
from tqdm import tqdm


def _gen_box_plots(scenario_dir: Path,
                  flatten_mode: bool = False,
                  input_data_fmt=dpr.DATA_FMTS['GPS'],
                  figsize=(4, 3),
                  plot_blotches: bool = False,
                  **kwargs) -> plt.Figure:

    # Get a list of paths to the stats.json files of each taxi.
    simulation_outputs_dir = scenario_dir.joinpath('EV_Results')
    ev_stats_files = sorted([
        *simulation_outputs_dir.glob('*/Outputs/stats_*.json')
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
            x = np.random.normal(loc=1 + i, scale=0.04, size=len(energy_diffs))
            plt.scatter(x, energy_diffs, alpha=0.4)

        ev_names.append('_'.join(ev_stats_file.stem.split('_')[1:]))

    # Calculating the distance travelled on each day.
    all_dists_travelled = []

    ev_dirs = sorted([
        *scenario_dir.joinpath('EV_Simulation',
            'EV_Simulation_Outputs').glob('*/')
    ])
    for ev_dir in tqdm(ev_dirs):
        dists_travelled = []
        ev_csv_files = sorted([
            *ev_dir.glob('*/battery.out.csv')])
        for ev_csv_file in ev_csv_files:
            ev_df = pd.read_csv(ev_csv_file, skipinitialspace=True)
            dist_travelled = ev_df['vehicle_speed'].sum()
            dists_travelled.append(dist_travelled)
        all_dists_travelled.append(dists_travelled)

    if input_data_fmt == dpr.DATA_FMTS['GTFS'] or flatten_mode:
        # Flatten the energy diffs array, so that only one box plot is created.
        all_energy_diffs_grouped = all_energy_diffs
        all_energy_diffs = []
        for energy_diffs in all_energy_diffs_grouped:
            all_energy_diffs.extend(energy_diffs)
        all_dists_travelled_grouped = all_dists_travelled
        all_dists_travelled = []
        for dists_travelled in all_dists_travelled_grouped:
            all_dists_travelled.extend(dists_travelled)
        ev_names_bak = ev_names
        ev_names = ['Fleet']
        kwh_p_km = (np.array(all_energy_diffs) /
                    (np.array(all_dists_travelled) / 1000))
    else:
        kwh_p_km = []
        for energy_diffs, dists_travelled in zip(all_energy_diffs,
                all_dists_travelled):
            kwh_p_km.append(np.array(energy_diffs) /
                            (np.array(dists_travelled) / 1000))

    # Generate a box-plot from the values in all_energy_diffs.
    plt.boxplot(kwh_p_km,
                medianprops={'color': 'black'},
                flierprops={'marker': '.'})
    plt.ylabel("Daily energy usage (kWh/km)")
    plt.xticks(range(1, len(ev_names) + 1), ev_names, rotation=30,
               fontsize='small')
    plt.xlabel("eMBT ID")
    plt.tight_layout()
    fig = plt.gcf()

    if input_data_fmt == dpr.DATA_FMTS['GTFS'] or flatten_mode:
        # Restore (or unflatten) the energy diffs array.
        all_energy_diffs = all_energy_diffs_grouped
        all_dists_travelled = all_dists_travelled_grouped
        ev_names = ev_names_bak

    export_data_zipped = []
    for i, ev_name in enumerate(ev_names):
        ev_name_list = [*repeat(ev_name, len(all_date_strs[i]))]
        export_data_zipped.extend(
            [*zip(ev_name_list, all_date_strs[i],
                  all_energy_diffs[i], all_dists_travelled[i])]
        )

    df = pd.DataFrame(export_data_zipped,
        columns=['ev_name', 'date', 'energy_used', 'dist_travelled'])
    df.set_index(['ev_name', 'date'])  # This is not necessary for the saving
        # process, but it's useful in-case this script is extended.

    return fig, df


def plot_ev_energy_boxes(scenario_dir: Path, **kwargs):

    _ = dpr.auto_input("Would you like to plot the box-plots of the fleet's "
                       "energy usage? [y]/n  ", 'y', **kwargs)
    if _.lower() == 'n':
        return

    # _ = dpr.auto_input("Would youl like to flatten the box-plots? y/[n]  ", 'n', **kwargs)
    # flatten_mode = False if _.lower() != 'y' else True

    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])
    box_plots_dir = scenario_dir.joinpath('EV_Results', 'Outputs', 'Box_Plots')

    print("Generating box-plots...")

    for flatten_mode in tqdm((False, True)):

        dir_name = 'flattened' if flatten_mode else 'non_flattened'
        box_plots_subdir = box_plots_dir.joinpath(dir_name)
        box_plots_subdir.mkdir(parents=True, exist_ok=True)

        fig, df = _gen_box_plots(scenario_dir, flatten_mode, input_data_fmt)

        # As png:
        fig_dir = box_plots_subdir.joinpath("Energy_usage_box_plots.png")
        fig.savefig(fig_dir)
        # As svg:
        fig_dir = box_plots_subdir.joinpath("Energy_usage_box_plots.pdf")
        fig.savefig(fig_dir)
        # As pickle:
        fig_dir = box_plots_subdir.joinpath(
            "Energy_usage_box_plots.fig.pickle")
        pickle.dump(fig, open(fig_dir, 'wb'))

        # Convert all_energy_diffs to a dataframe and save as a csv file.
        csv_dir = box_plots_subdir.joinpath("Energy_usage.csv")
        df.to_csv(csv_dir, index=False)

    auto_run = kwargs.get('auto_run', False)
    if not auto_run:
        plt.show()
