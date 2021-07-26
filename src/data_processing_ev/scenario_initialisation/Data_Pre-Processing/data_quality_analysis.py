#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from haversine import haversine  # Used for calculating distance between GPS
                                 # coordinates.
import pickle


def __do_calc_samples(vehicle_file: Path) -> pd.DataFrame:
    vehicle_df = pd.read_csv(vehicle_file)
    vehicle_df['Time'] = pd.to_datetime(vehicle_df['Time'])
    dates_series = vehicle_df['Time'].map(lambda t: t.date())
    samples_per_day = dates_series.value_counts().sort_index()
    samples_df = samples_per_day.reset_index()
    samples_df.columns = ['Date', 'Count']
    index = pd.date_range(min(samples_df['Date']),
                          max(samples_df['Date']))
    samples_df = samples_df.set_index(['Date'])
    samples_df = samples_df.reindex(index, fill_value=0)
    return samples_df


def _plot_samples_per_day(traces_dir: Path):
    """Plot the samples per day as 10 line plots. (1 for each taxi.) """
    print("Calculating samples per day for each taxi...")
    vehicle_files = sorted([*traces_dir.joinpath('Processed').glob('*.csv')])
    ev_names = [vehicle_file.stem for vehicle_file in vehicle_files]
    fig, ax = plt.subplots()
    args = zip(vehicle_files)

    with Pool(10) as p:
        sample_dataframes = p.starmap(__do_calc_samples, args)

    # Non Pooled version:
    # sample_dataframes = itertools.starmap(__do_calc_samples, args)

    for sample_dataframe in sample_dataframes:
        ax.plot(sample_dataframe)
    ax.set_title("Samples recorded per day for each vehicle")
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("Date")
    ax.legend(ev_names)
    save_file = traces_dir.joinpath("Graphs", "samples_per_day.pdf")
    save_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file)
    save_file = traces_dir.joinpath("Graphs", "samples_per_day.png")
    plt.savefig(save_file, dpi=300)
    save_file = traces_dir.joinpath("Graphs", "samples_per_day.fig.pickle")
    pickle.dump(plt.gcf(), open(save_file, 'wb'))
    plt.show()
    return


def __do_calc_max_spatial_dist(vehicle_file: Path) -> pd.DataFrame:
    vehicle_df = pd.read_csv(vehicle_file)
    vehicle_df['Time'] = pd.to_datetime(vehicle_df['Time'])
    distances = [None]
    for window in vehicle_df.rolling(window=2):
        if len(window) < 2:
            continue
        distance = haversine(window.iloc[0].loc[['Latitude', 'Longitude']],
                             window.iloc[1].loc[['Latitude', 'Longitude']],
                             unit='m')
        distances.append(distance)
    distances_df = vehicle_df[['Time']]
    distances_df['Distance'] = distances
    distances_df['Time'] = distances_df['Time'].map(lambda t: t.date())
    distances_df.columns = ['Date', 'Distance']
    date_index = pd.date_range(min(distances_df['Date']),
                               max(distances_df['Date']))
    distances_df = distances_df.groupby(['Date']).max()
    distances_df = distances_df.reindex(date_index, fill_value=0)
    return distances_df


def _plot_max_spatial_jump_per_day(traces_dir: Path) -> None:
    print("Calculating max spatial jump per day for each taxi...")
    vehicle_files = sorted([*traces_dir.joinpath('Processed').glob('*.csv')])
    ev_names = [vehicle_file.stem for vehicle_file in vehicle_files]
    args = zip(vehicle_files)

    with Pool(10) as p:
        distance_dfs = p.starmap(__do_calc_max_spatial_dist, args)

    # Non Pooled version:
    # distance_dfs = [*itertools.starmap(__do_calc_max_spatial_dist, args)]

    [*map(plt.plot, distance_dfs)]
    plt.title("Maximum spatial jump per day fro each vehicle")
    plt.ylabel("Maximum distance (m)")
    plt.xlabel("Date")
    plt.legend(ev_names)

    save_file = traces_dir.joinpath("Graphs", "max_spatial_jump_per_day.pdf")
    save_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file)
    save_file = traces_dir.joinpath("Graphs", "max_spatial_jump_per_day.png")
    plt.savefig(save_file, dpi=300)
    save_file = traces_dir.joinpath("Graphs",
                                    "max_spatial_jump_per_day.fig.pickle")
    pickle.dump(plt.gcf(), open(save_file, 'wb'))
    plt.show()
    return


def main(scenario_dir: Path):
    # Create a list of csv files found in the traces directory.
    traces_dir = scenario_dir.joinpath('_Inputs', 'Traces')

    if not traces_dir.joinpath('Graphs', 'samples_per_day.pdf').exists():
        _plot_samples_per_day(traces_dir)
    else:
        _ = input("(Re)generate samples_per_day plot? [y/N] ")
        if _.lower() == 'y':
            _plot_samples_per_day(traces_dir)

    if not traces_dir.joinpath('Graphs',
                               'max_spatial_jump_per_day.pdf').exists():
        _plot_max_spatial_jump_per_day(traces_dir)
    else:
        _ = input("(Re)generate max_spatial_jump_per_day plot? [y/N] ")
        if _.lower() == 'y':
            _plot_max_spatial_jump_per_day(traces_dir)


if __name__ == '__main__':
    scenario_dir = Path(os.path.abspath(__file__)).parents[2]  # XXX This isn't working when pdb is loaded...
    main(scenario_dir)
