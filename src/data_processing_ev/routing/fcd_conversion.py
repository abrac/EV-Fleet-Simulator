#!/usr/bin/env python3

"""
This script allows the routing steps to be skipped. This is useful for 1 Hz
data. If routing is skipped, the resulting FCD output needs to be processed
with the Hull et al. EV model.
"""

import os
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from operator import itemgetter
import numpy as np
import datetime as dt
import data_processing_ev as dpr

MAX_VELOCITY = np.inf  # km/h


def _reformat_time(time_old: str) -> str:
    # time format: "year/month/day hr12:min:sec meridiem" -->
    # "year-month-day hr24:min:sec"
    (date, time, meridiem) = time_old.split(' ')
    (year, month, day) = date.split("/")
    (hr12, minute, sec) = time.split(":")
    hr24 = (int(hr12) % 12) + (12 if meridiem == "PM" else 0)
    return f"{year}-{month}-{day} {hr24:02d}:{minute}:{sec}"


def _generate_traces(traces_dir: Path):
    # For each file, identify the vehicle_id corresponding to it and the
    # start-date of the file's data.
    original_files = sorted([*traces_dir.joinpath('Original').glob('*.csv')])

    # For each vehicle_id:
    for original_file in tqdm(original_files):
        output_file = traces_dir.joinpath('Processed',
                                          original_file.name)
        # Write the header row.
        header_row = ("GPSID,Time,Latitude,Longitude,Altitude," +
                      "Heading,Satellites,HDOP,AgeOfReading,"   +
                      "DistanceSinceReading,Velocity")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f_out:
            f_out.write(header_row + '\n')
        # For each file in files_of_vehicle:
        new_rows = []
        with open(original_file, 'r') as f_in:
            # Read and discard the header row.
            f_in.readline()
            # For each remaining row in the file:
            for row in f_in:
                # Re-format the row.
                row_vals = row[:-1].split(',')
                velocity = float(row_vals[10])
                if velocity < MAX_VELOCITY:
                    new_row_vals = []
                    new_row_vals.append(row_vals[0])  # GPSID
                    new_row_vals.append(_reformat_time(row_vals[1]))  # Time
                    new_row_vals.append(row_vals[2])  # Latitude
                    new_row_vals.append(row_vals[3])  # Longitude
                    new_row_vals.append(row_vals[4])  # Altitude
                    new_row_vals.append(row_vals[5])  # Heading
                    new_row_vals.append(row_vals[6])  # Satellites
                    new_row_vals.append(row_vals[7])  # HDOP
                    new_row_vals.append(row_vals[8])  # AgeOfReading
                    new_row_vals.append(row_vals[9])  # DistanceSinceReading (km)
                    new_row_vals.append(row_vals[10])  # Velocity (mph -> km/h)
                    new_rows.append(new_row_vals)

        # Write the rows in the vehicle's output file.
        new_rows = sorted(new_rows, key=itemgetter(1))  # Sort rows by time.
        new_rows = [','.join(row) + '\n' for row in new_rows]
        with open(output_file, 'a') as f_out:
            # f_out.write(new_rows + '\n')
            f_out.writelines(new_rows)


def convert_data(scenario_dir: Path, **kwargs):
    breakpoint()  # XXX

    # List the folders containing the traces of each EV.
    ev_dirs = scenario_dir.joinpath(
        'Spatial_Clusters', 'Filtered_Traces').glob('*/')

    # For each EV, convert its traces to a SUMO FCD-style output.
    for ev_dir in ev_dirs:
        for trace_path in ev_dir.glob('*.csv'):
            trace = pd.read_csv(trace_path)
            trace_new = pd.DataFrame()
            first_time: dt.datetime = trace['Time'][0]
            year, month, day = first_time.date.split('-')
            midnight = dt.datetime(year, month, day)
            trace_new['timestep_time'] = trace['Time'].\
                map(lambda time: (time - midnight).second)
            trace_new['vehicle_speed'] = trace['Velocity']
            trace_new['vehicle_x'] = trace['Longitude']
            trace_new['vehicle_y'] = trace['Latitude']
            altitude_conversion = None
            if 'Altitude' in trace.columns:
                if altitude_conversion is None:
                    _ = dpr.auto_input(
                        "Altitude data found in the input data. Would you "
                        "like to use them in the EV simulations? ([y]/n)  ",
                        'y', **kwargs)
                    altitude_conversion = True if _.lower() != 'n' else False
                if altitude_conversion:
                    trace_new['vehicle_z'] = trace['Altitude']
            output_file = scenario_dir.joinpath(
                'Routes', 'FCD_Data', ev_dir.name,
                trace_path.stem.split('_')[1], 'fcd.out.csv')
            trace_new.to_csv(output_file)


if __name__ == '__main__':
    scenario_dir = Path(os.path.abspath(__file__))
    convert_data(scenario_dir, kwargs={'auto_run': False})
