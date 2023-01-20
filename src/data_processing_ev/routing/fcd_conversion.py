#!/usr/bin/env python3

"""
This script allows the routing steps to be skipped. This is useful for 1 Hz
data. If routing is skipped, the resulting FCD output needs to be processed
with the Hull et al. EV model.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import datetime as dt
import data_processing_ev as dpr

MAX_VELOCITY = np.inf  # km/h


def convert_data(scenario_dir: Path, **kwargs):
    # List the folders containing the traces of each EV.
    ev_dirs = sorted([*scenario_dir.joinpath(
        'Spatial_Clusters', 'Filtered_Traces').glob('*/')])

    altitude_conversion = None
    # For each EV, convert its traces to a SUMO FCD-style output.
    for ev_dir in ev_dirs:
        for trace_path in sorted([*ev_dir.glob('*.csv')]):
            trace = pd.read_csv(trace_path)

            # Check if the trace has duplicate timestamps!
            duplicates = trace['Time'].duplicated()
            if True in duplicates.values:
                dpr.LOGGERS['main'].error(
                    "The input trace consists of duplicate time steps. "
                    "*Discarding* duplicates. Manually correct these in the "
                    "input trace and re-run EV-Fleet-Sim from the spatial "
                    f"clustering step. Duplicate values in {trace_path} "
                    f"are: \n{trace[duplicates]}.")
                trace = trace[~duplicates]

            trace_new = pd.DataFrame()
            trace['Time'] = trace['Time'].astype('datetime64')
            first_time: dt.datetime = trace['Time'][0]
            date = first_time.date()
            midnight = dt.datetime(date.year, date.month, date.day)
            trace_new['timestep_time'] = trace['Time'].\
                map(lambda time: (time - midnight).seconds)
            trace_new['vehicle_speed'] = trace['Velocity']/3.6
                # FIXME Don't convert km/h to m/s! The input should
                # already be in m/s.
            trace_new['vehicle_x'] = trace['Longitude']
            trace_new['vehicle_y'] = trace['Latitude']
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
                'Mobility_Simulation', 'FCD_Data', ev_dir.name,
                trace_path.stem.split('_')[-1], 'fcd.out.csv')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            trace_new.to_csv(output_file)
            dpr.compress_file(output_file)


if __name__ == '__main__':
    scenario_dir = Path(os.path.abspath(__file__))
    convert_data(scenario_dir, kwargs={'auto_run': False})
