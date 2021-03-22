#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from operator import itemgetter
import matplotlib.pyplot as plt
from multiprocessing import Pool
from haversine import haversine  # Used for calculating distance between GPS
                                 # coordinates.
import pickle

MAX_VELOCITY = 2  # km/h


def _generate_traces(traces_dir: Path):
    # For each file, identify the vehicle_id corresponding to it and the
    # start-date of the file's data.
    original_files = sorted([*traces_dir.joinpath('Original').glob('*/*.csv')])
    vehicle_ids = []
    start_dates = []
    for original_file in original_files:
        with open(original_file, 'r') as f:
            f.readline()
            row = f.readline().split(',')
            vehicle_id = int(row[1][1:-1])
            start_date = row[9].split(' ')[0][1:]
        vehicle_ids.append(vehicle_id)
        start_dates.append(start_date)
    # Data-frame of the three arrays:
    original_data = pd.DataFrame({
        'vehicle_id': vehicle_ids,
        'start_date': start_dates,
        'original_file': original_files
    })
    original_data['start_date'] = pd.to_datetime(
        original_data.loc[:, 'start_date'])
    original_data = original_data.set_index(
        ['vehicle_id', 'start_date']).sort_index()

    # For each vehicle_id:
    for vehicle_id in tqdm(
            set(original_data.index.get_level_values('vehicle_id'))):
        output_file = traces_dir.joinpath('Processed',
                                          f'T{vehicle_id:0=2}.csv')
        # Write the header row.
        header_row = ("GPSID,Time,Latitude,Longitude,Altitude," +
                      "Heading,Satellites,HDOP,AgeOfReading,"   +
                      "DistanceSinceReading,Velocity")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f_out:
            f_out.write(header_row + '\n')
        # For each file in files_of_vehicle:
        new_rows = []
        for file in original_data.loc[vehicle_id]['original_file']:
            with open(file, 'r') as f_in:
                # Read and discard the header row.
                f_in.readline()
                # For each remaining row in the file:
                for row in f_in:
                    # Re-format the row.
                    row_vals = [row_val[1:-1] for row_val in row.split(',')]
                    velocity = float(row_vals[8]) * 1.609344
                    if velocity < MAX_VELOCITY:
                        xml_info = ET.fromstring(row_vals[6])  # Column 6 is
                            # extra xml data.
                        new_row_vals = []
                        new_row_vals.append(row_vals[0])  # GPSID
                        new_row_vals.append(row_vals[9])  # Time
                        new_row_vals.append(row_vals[4])  # Latitude
                        new_row_vals.append(row_vals[5])  # Longitude
                        new_row_vals.append(row_vals[2])  # Altitude
                        new_row_vals.append('')           # Heading
                        new_row_vals.append(xml_info.find('sat').text)   # Satellites
                        new_row_vals.append(xml_info.find('hdop').text)  # HDOP
                        new_row_vals.append('')  # AgeOfReading
                        new_row_vals.append(
                            str(float(row_vals[12]) * 1.609344))
                            # DistanceSinceReading (mi -> km)
                        new_row_vals.append(str(velocity))  # Velocity (mph -> km/h)
                        new_rows.append(new_row_vals)

        # Write the row in the vehicle's output file.
        new_rows = sorted(new_rows, key=itemgetter(1))  # Sort rows by time.
        new_rows = [','.join(row) + '\n' for row in new_rows]
        with open(output_file, 'a') as f_out:
            # f_out.write(new_rows + '\n')
            f_out.writelines(new_rows)


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

    if not traces_dir.joinpath('Processed').glob('*.csv'):
        _generate_traces(traces_dir)
    else:
        # Else (if there are aleardy processed csv files):
        _ = input("(Re)generate processed traces? [y/N] ")
        if _.lower() == 'y':
            _generate_traces(traces_dir)

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
    scenario_dir = Path(os.path.abspath(__file__)).parents[2]
    main(scenario_dir)
