#!/usr/bin/env python3

"""
This script generates a dataframe which contains the following format:

|-------+------------+--------------+---------------------|
| Taxi  | Date       | Stop Arrival | Stop Duration (hrs) |
|-------+------------+--------------+--------------------:|
| T1000 | 2019-04-03 | 16:39        |                 3.2 |
| ⋮     | ⋮          | ⋮            |                   ⋮ |
|-------+------------+--------------+---------------------|
"""

# FIXME Incorporate this script into temporal_clustering.py in the code
    # repository.

import os
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Iterator
import datetime as dt
from itertools import repeat
from tqdm import tqdm


def _time_str2dt(datetime_str: str) -> dt.datetime:
    date, time = datetime_str.split()
    hr24, minute, sec = [int(time_old) for time_old in
                         time.split(':')]
    date_arr = [int(date_part) for date_part in date.split('-')]
    return dt.datetime(*date_arr, hr24, minute, sec)

    # date, time, meridiem = datetime_str.split()
    # hr12, minute, sec = [int(time_old) for time_old in
    #                      time.split(':')]
    # hr24 = (hr12 % 12) + (12 if meridiem == "PM" else 0)
    # date_arr = [int(date_part) for date_part in date.split('/')]
    # return dt.datetime(*date_arr, hr24, minute, sec)

def _gen_trace_dfs(scenario_dir: Path) -> Iterator[Tuple[pd.DataFrame, str]]:
    """
    Generate dataframes of ordered stop-clusters for each EV in the scenario.
    """
    # Get the location of the filtered traces.
    traces_root_dir = scenario_dir.joinpath('Spatial_Clusters',
                                            'Filtered_Traces')
    # create an array of their directories, corresponding to each EV.
    trace_dirs = [f for f in traces_root_dir.iterdir() if f.is_dir()]
    # If there are no traces, throw an error.
    if len(trace_dirs) < 1:
        raise ValueError('No traces found in \n\t{0}.'.format(
            traces_root_dir.absolute()))
    # For each EV yield a dataframe.
    for trace_dir in trace_dirs:
        # Get the trace files and sort them by date.
        trace_files = sorted(trace_dir.glob('*.csv'))
        # Convert the traces to dataframes and append them to a list.
        trace_dfs = []
        for trace_file in trace_files:
            trace_dfs.append(pd.read_csv(trace_file))
        # Convert the list of dataframes to huge dataframe.
        trace_df = pd.concat(trace_dfs, ignore_index=True)
        # Yield it the dataframe, along with the EV's name.
        yield (trace_df, trace_dir.stem)


def _build_stops_df(trace_dfs_generator: Iterator[Tuple[pd.DataFrame, str]]
                    ) -> pd.DataFrame:
    """Generate the dataframe of stop-arrivals and -durations."""

    stops_dfs = []

    # Generate the dataframe of stop-arrivals and -durations for each EV, and
        # append to trace_dfs.

    print("FIXME: Increase this threshod to around 10 km/h! \n" +
          "Don't filter out datapoints based on speed in the " +
          "data-pre-processing step!")
    _ = input("Press enter to acknowledge the above warning.")
    for trace_df, ev_name in tqdm(trace_dfs_generator):

        # First: Generate a list of stop-entry and exit times.
        stop_entries_and_exits: List[Tuple[pd.Series, int]] = []
        stop_encountered = False
        prev_datapoint = None
        entry_datapoint = None
        start_new_entry = True
        for _, datapoint in trace_df.iterrows():
            # Check for consecutive points that have at least one point where
            # ... the taxi was stopped.
            # Continue until the taxi moved at a speed greater than or equal to
            # ... 1 km/h.
            # FIXME: Increase this threshod to around 10 km/h! Don't filter out
            # ... datapoints based on speed in the data-pre-processing step!
            if datapoint['Velocity'] >= 1:
                start_new_entry = True
            # If the taxi left the space cluster, or this is the first
            #  iteration...
            if start_new_entry:
                # If a stop was encountered between the entry datapoint and the
                # previous datapoint (i.e. just before the taxi left the
                # spatial cluster)...
                if stop_encountered:
                    # Record entry_datapoint and exit_datapoint
                    stop_entries_and_exits.append(
                        (entry_datapoint, prev_datapoint))
                # Reset the flags
                start_new_entry = False
                stop_encountered = False
                # Make the datapoint after the jump, the new entry datapoint
                entry_datapoint = datapoint
            if datapoint['Velocity'] < 1:
                stop_encountered = True
            prev_datapoint = datapoint

        # If list of stop_times still empty, throw exception, and continue to
            # next EV.
        if stop_entries_and_exits == []:
            print(f"Error: no stops found for EV {ev_name}.")
            continue

        # Secondly: Calulate stop-durations from stop-entry and -exit times.
        stop_durations = []
        for (stop_entry, stop_exit) in stop_entries_and_exits:
            entry_time = _time_str2dt(stop_entry['Time'])
            exit_time = _time_str2dt(stop_exit['Time'])
            stop_duration = (exit_time-entry_time).total_seconds()/3600
            if stop_duration < 0:
                raise ValueError("Stop duration was negative!")
            stop_durations.append(stop_duration)
        # Convert stop-arrival times from date-time objects to floats
            # which represent the hour arrived.
        stop_arrivals = [
            dt.timedelta(hours=entry_time.hour, minutes=entry_time.minute,
                         seconds=entry_time.second).total_seconds()/3600 for
            entry_time in [_time_str2dt(stop_entry['Time']) for
                           (stop_entry, _) in stop_entries_and_exits]]
            # TODO: Maybe I should keep stop_arrivals as dt objects? Why
                # convert and lose information?
        dates = [
            entry_datetime.date() for
            entry_datetime in [_time_str2dt(stop_entry['Time']) for
                           (stop_entry, _) in stop_entries_and_exits]]
        cluster_nums = [stop_entry['Cluster'] for (stop_entry, _) in
                        stop_entries_and_exits]

        # Combine stop_arrivals and stop_durations into one dataframe.
        stops_df = pd.DataFrame(
            zip(
                dates, stop_arrivals, stop_durations, cluster_nums
            ),
            columns=['Date', 'Stop_Arrival', 'Stop_Duration', 'Cluster']
        )

        # Thirdly: Remove stop-events that are irrelevant.
        def _filter_stop_events(stops_df: pd.DataFrame, dropping=True):
            # TODO Make these function arguments.
            max_stop_duration = 8
            min_stop_duration = 0.33
            min_stop_arrival = 0 # XXX Revert
            max_stop_arrival = 24  # XXX Revert
            # Create new column, which says whether a row is invalid or not
            stops_df['Invalid'] = False
            # Create dataframe, mask, which says whether any of the four bad
            # conditions are met at any row
            mask = (
                (stops_df['Stop_Duration'] < min_stop_duration) |
                (stops_df['Stop_Duration'] > max_stop_duration) |
                (stops_df['Stop_Arrival'] < min_stop_arrival) |
                (stops_df['Stop_Arrival'] > max_stop_arrival)
            )
            # Mark invalid True for all rows which are True in the mask
            stops_df.loc[mask, 'Invalid'] = True
            # Optionally drop rows for which 'Invalid' is True.
            if dropping:
                stops_df = stops_df.drop(
                    stops_df[stops_df['Invalid']].index
                )
                # drop 'Invalid' column
                stops_df = stops_df.drop('Invalid', axis=1)
            return(stops_df)
        stops_df = _filter_stop_events(stops_df, dropping=True)
        # If all the stops were filtered out, throw warning, and continue.
        if stops_df.empty:
            print(f"Warning: all stops filtered out for EV {ev_name}.")  # TODO Change to logging.warn # noqa
            continue

        # Add "ev_name" as a columns to the dataframe, and then set "ev_name" &
            # "date" as indices.
        stops_df['EV_Name'] = ev_name
        stops_df = stops_df.set_index(['EV_Name', 'Date'])

        # Finally: Append the data frame and the ev_name to the lists.
        stops_dfs.append(stops_df)

    # Combine the dataframes in the lists into one.
    full_stops_df = pd.concat(stops_dfs)
    return full_stops_df


def main():
    scenario_dir = Path(os.path.abspath(__file__)).parents[2]  # FIXME Make this relative to
        # scenario_dir.
    stops_output_file = Path(__file__).parent.joinpath(
        'dirty__time_clusters_with_dates.csv'
    )  # FIXME  Make this relative to scenario_dir.
    trace_dfs_generator = _gen_trace_dfs(scenario_dir)
    stops_df = _build_stops_df(trace_dfs_generator)
    stops_df.to_csv(stops_output_file)


if __name__ == "__main__":
    main()
