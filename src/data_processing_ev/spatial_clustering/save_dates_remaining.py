#!/usr/bin/env python3
"""
This script generates a csv file which records the dates which have survived
the filtering process.
"""

import os
from pathlib import Path
import datetime as dt
import csv
import data_processing_ev as dpr


def save_dates_remaining(scenario_dir: Path, **kwargs):

    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])

    # Get a list of EV_Names.
    filtered_traces_dir = scenario_dir.joinpath('Spatial_Clusters',
                                                'Filtered_Traces')
    ev_names = sorted(next(os.walk(filtered_traces_dir))[1])

    dates = []
    # For each EV, get a list of dates, and create a df for the EV.
    for ev_name in ev_names:
        ev_dir = filtered_traces_dir.joinpath(ev_name)
        filtered_files = sorted([*ev_dir.glob('*.csv')])

        if input_data_fmt == dpr.DATA_FMTS['GPS']:
            for filtered_file in filtered_files:
                year, month, day = filtered_file.stem.split('_')[-1].split('-')
                date = dt.date(int(year), int(month), int(day))
                dates.append((ev_name, date))
        elif input_data_fmt == dpr.DATA_FMTS['GTFS']:
            for filtered_file in filtered_files:
                day = int(filtered_file.stem.split('_')[-1][:-1])
                dates.append((ev_name, day))
        else:
            raise ValueError(dpr.DATA_FMT_ERROR_MSG)

    # Save the dates as a csv file.
    header = ('EV_Name', 'Date')
    with open(filtered_traces_dir.joinpath('dirty__dates_remaining.csv'),
              'w') as f:
        wtr = csv.writer(f, delimiter=',', lineterminator='\n')
        wtr.writerow(header)
        wtr.writerows(dates)


if __name__ == '__main__':
    scenario_dir = Path(os.path.abspath(__file__)).parents[2]
    save_dates_remaining(scenario_dir)
