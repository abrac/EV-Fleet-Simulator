#!/usr/bin/env python3

import os
from pathlib import Path
from tqdm import tqdm
import math

def get_max_min_coords(traces_dir: Path):
    # For each file, identify the vehicle_id corresponding to it and the
    # start-date of the file's data.
    processed_files = sorted([*traces_dir.joinpath('Processed').glob('*.csv')])

    max_latitude = -math.inf
    max_longitude = -math.inf
    min_latitude = math.inf
    min_longitude = math.inf

    # For each vehicle_id:
    for processed_file in tqdm(processed_files):
        output_file = traces_dir.joinpath('Processed',
                                          processed_file.name)
        with open(processed_file, 'r') as f_in:
            # Read and discard the header row.
            f_in.readline()
            # For each remaining row in the file:
            for row in f_in:
                # Re-format the row.
                row_vals = row[:-1].split(',')
                latitude = float(row_vals[2])
                longitude = float(row_vals[3])

                if latitude > max_latitude:
                    max_latitude = latitude
                if latitude < min_latitude:
                    min_latitude = latitude
                if longitude > max_longitude:
                    max_longitude = longitude
                if longitude < min_longitude:
                    min_longitude = longitude

    return {'min_longitude': min_longitude, 'max_longitude': max_longitude,
            'min_latitude': min_latitude, 'max_latitude': max_latitude}


if __name__ == '__main__':
    traces_dir = Path(os.path.abspath(__file__)).parent
    print(get_max_min_coords(traces_dir))
