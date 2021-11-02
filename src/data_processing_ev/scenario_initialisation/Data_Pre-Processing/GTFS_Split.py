#!/usr/bin/env python3

import os
import pandas as pd
from pathlib import Path
import datetime as dt
from tqdm import tqdm

def main(scenario_dir: Path):

    def _reformat_time(cumtime: int) -> str:

        timedelta = dt.timedelta(seconds=cumtime)
        days = timedelta.days
        hms = str(timedelta).split(', ')[-1]
        hour, minute, second = hms.split(':')
        new_hour = int(hour)  # * 24
        new_time = f'{days}d {new_hour:02d}:{minute}:{second}'

        return new_time

    monolithic_file = scenario_dir.joinpath('_Inputs', 'Traces', 'Processed',
                                            'monolithic', 'traces_orig.csv')
    processed_traces_dir = monolithic_file.parents[1]

    gps_dfs = [*pd.read_csv(monolithic_file, dtype=str).\
               groupby(['shape_id', 'trip_id', 'trip_number'])]

    print("Calculating the boundaries of the scenario...")
    inf = float('inf')
    lon_limits = {'min': inf, 'max': -inf}
    lat_limits = {'min': inf, 'max': -inf}
    for _, gps_df in tqdm(gps_dfs):
        min_lon = min(gps_df['shape_pt_lon'].astype('float'))
        max_lon = max(gps_df['shape_pt_lon'].astype('float'))
        min_lat = min(gps_df['shape_pt_lat'].astype('float'))
        max_lat = max(gps_df['shape_pt_lat'].astype('float'))
        if min_lon < lon_limits['min']:
            lon_limits['min'] = min_lon
        if min_lat < lat_limits['min']:
            lat_limits['min'] = min_lat
        if max_lon > lon_limits['max']:
            lon_limits['max'] = max_lon
        if max_lat > lat_limits['max']:
            lat_limits['max'] = max_lat

    input("The datapoints can be contained within the following " +
          f"boundaries:\n\tLongitude: {lon_limits}\n\tLatitude: {lat_limits}" +
          "\nPress enter to continue...")

    for (_, trip_id, _), gps_df in gps_dfs:

        if trip_id.find('#') == -1:
            trip_id = trip_id
        else:
            trip_id = '#'.join(trip_id.split('#')[:-1])  # Remove the trip
                                                         # number from the trip
                                                         # ID.

        # Remove special characters from trip_id.
        trip_id = trip_id.replace(' ', '_').replace('-', '_').\
            replace('(', '').replace(')', '')

        output_file = processed_traces_dir.joinpath(f"{trip_id}.csv")

        if output_file.exists():
            continue

        gps_df = gps_df.fillna('')

        # Write the header row.
        header_row = ("GPSID,Time,Latitude,Longitude,Altitude," +
                      "Heading,Satellites,HDOP,AgeOfReading,"   +
                      "DistanceSinceReading,Velocity,StopID")

        with open(output_file, 'w') as f_out:
            f_out.write(header_row + '\n')

        # For each row in the trip dataframe:
        new_rows = []
        for _, row in gps_df.iterrows():
            new_row = []
            new_row.append(row['id'])  # GPSID
            new_row.append(_reformat_time(int(float(row['cumtime']))))  # Time
            new_row.append(row['shape_pt_lat'])  # Latitude
            new_row.append(row['shape_pt_lon'])  # Longitude
            new_row.append('')  # Altitude
            new_row.append('')  # Heading
            new_row.append('')  # Satellites
            new_row.append('')  # HDOP
            new_row.append('')  # AgeOfReading
            new_row.append(row['dist'])  # DistanceSinceReading
            new_row.append(row['speed'])  # Velocity
            new_row.append(row['stop_id'])  # StopID
            new_rows.append(new_row)

        new_rows = [','.join(row) + '\n' for row in new_rows]

        with open(output_file, 'a') as f_out:
            f_out.writelines(new_rows)


if __name__ == "__main__":
    scenario_dir = Path(os.path.abspath(__file__)).parents[2]
    main(scenario_dir)
