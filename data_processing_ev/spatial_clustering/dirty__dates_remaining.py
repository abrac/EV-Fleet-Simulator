#!/usr/bin/env python3
"""
This script generates a csv file which records the dates which have survived
the filtering process.
"""

import os
from pathlib import Path
import datetime as dt
import csv


def main(scenario_dir: Path):
    # Get a list of EV_Names.
    filtered_traces_dir = scenario_dir.joinpath('Spatial_Clusters',
                                                'Filtered_Traces')
    ev_names = sorted(next(os.walk(filtered_traces_dir))[1])

    dates = []
    # For each EV, get a list of dates, and create a df for the EV.
    for ev_name in ev_names:
        ev_dir = filtered_traces_dir.joinpath(ev_name)
        filtered_files = sorted([*ev_dir.glob('*.csv')])

        for filtered_file in filtered_files:
            year, month, day = filtered_file.stem.split('_')[1].split('-')
            date = dt.date(int(year), int(month), int(day))
            dates.append((ev_name, date))

    # Save the dates as a csv file.
    header = ('EV_Name', 'Date')
    with open(filtered_traces_dir.joinpath('dirty__dates_remaining.csv'),
              'w') as f:
        wtr = csv.writer(f, delimiter=',', lineterminator='\n')
        wtr.writerow(header)
        wtr.writerows(dates)


if __name__ == '__main__':
    scenario_dir = Path(os.path.abspath(__file__)).parents[2]
    main(scenario_dir)
