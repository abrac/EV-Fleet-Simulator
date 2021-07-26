#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from operator import itemgetter
import numpy as np

MAX_VELOCITY = np.inf  # km/h


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


if __name__ == '__main__':
    scenario_dir = Path(os.path.abspath(__file__)).parents[2]  # XXX This isn't working when pdb is loaded...
    main(scenario_dir)
