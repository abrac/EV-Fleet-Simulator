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
from haversine import haversine


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
    trace_dirs = sorted([f for f in traces_root_dir.iterdir() if f.is_dir()])
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


def _get_stop_entries_and_exits(trace_df: pd.DataFrame
        ) -> List[Tuple[pd.Series,
                        pd.Series]]:
    stop_entries_and_exits = []
    prev_datapoint = None
    entry_datapoint = None
    start_new_entry = False
    stop_encountered = False
    stop_location = None  # Coordinates of the first datapoint in the stop.
    for _, datapoint in trace_df.iterrows():
        # Check for consecutive points that have at least one point where
        # ... the taxi was stopped.
        # Continue until the taxi moved at a speed greater than or equal to
        # ... 10 km/h or the taxi drifts 25m from the stop location.
        if stop_encountered:
            current_location = (datapoint['Latitude'],
                                datapoint['Longitude'])
            distance_drifted = haversine(current_location, stop_location,
                                         unit='m')
            if datapoint['Velocity'] >= 10 or distance_drifted >= 25:
                start_new_entry = True
        # If the taxi stopped stopping:
        if start_new_entry:
            # # If a stop was encountered between the entry datapoint and the
            # # previous datapoint (i.e. just before the taxi left the
            # # spatial cluster)...
            # if prev_datapoint is not entry_datapoint:
            #     # Record entry_datapoint and exit_datapoint
            #     stop_entries_and_exits.append(
            #         (entry_datapoint, prev_datapoint))

            # Record the entry_datapoint and the current datapoint. We are
            # considering the current datapoint's timestamp to be the *end*
            # of the stop-event -- otherwise, single-datapoint stop-events
            # will be lost...
            stop_entries_and_exits.append(
                (entry_datapoint, datapoint))
            # Reset the flags
            start_new_entry = False
            stop_encountered = False
        if datapoint['Velocity'] < 1:
            # if this is the first stop that was encountered, make it the
            # "entry" datapoint of the stop event and record its location.
            if not stop_encountered:
                entry_datapoint = datapoint
                stop_location = (datapoint['Latitude'],
                                 datapoint['Longitude'])
            stop_encountered = True
        prev_datapoint = datapoint

    # If, after the loop, there is a stop_entry remaining without a stop_exit:
    if stop_encountered:
        stop_entries_and_exits.append((entry_datapoint, datapoint))

    return stop_entries_and_exits


def _build_stops_df(trace_dfs_generator: Iterator[Tuple[pd.DataFrame, str]]
                    ) -> Tuple[pd.DataFrame,
                               List[Tuple[str, pd.Series, pd.Series]]]:
    """Calculate the stops in the GPS traces.

    Returns:
        Tuple of:
            - Dataframe of stop-arrivals and -durations.
            - List of stop entries and exits.
    """

    stops_dfs_filtered = []

    # Generate the dataframe of stop-arrivals and -durations for each EV, and
        # append to trace_dfs.

    print("FIXME: Make sure that the stop-extraction code in this script is "
          "in sync with the code in the temporal-clustering script.")
    _ = input("Press enter to acknowledge the above warning.")
    for trace_df, ev_name in tqdm(trace_dfs_generator):

        # First: Generate a list of stop-entry and exit times.
        stop_entries_and_exits = _get_stop_entries_and_exits(trace_df)

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

        stops_df = pd.DataFrame(
            zip(
                dates, stop_arrivals, stop_durations, cluster_nums
            ),
            columns=['Date', 'Stop_Arrival', 'Stop_Duration', 'Cluster']
        )

        # Thirdly: Remove stop-events that are irrelevant.
        def _filter_stop_events(stops_df: pd.DataFrame, dropping=True):
            # TODO Make these function arguments.
            max_stop_duration = 8  # TODO Should this not be set to 24 hours?
                # I.e. Set to *no* limit?
            min_stop_duration = 0.33
            min_stop_arrival = 0  # TODO Revert to 6 AM.
            max_stop_arrival = 24  # TODO Revert to 6 PM.
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
        stops_dfs_filtered.append(stops_df)

    # Combine the dataframes in the lists into one.
    full_stops_df_filtered = pd.concat(stops_dfs_filtered)

    return full_stops_df_filtered


def _build_stop_labels(
            trace_dfs_generator: Iterator[Tuple[pd.DataFrame, str]]
        ) -> List[Tuple[str, pd.DataFrame]]:
    """
    Build stop labels corresponding to each datpoint in each EV.
    """

    all_stop_labels: List[pd.DataFrame] = []  # The stop_labels of each EV will
        # be appended onto this list.

    for (trace_df, ev_name) in tqdm(trace_dfs_generator):

        # Get the pairs of entries and exits for this trace.
        stop_pairs = _get_stop_entries_and_exits(trace_df)

        # An iterator to go through the stop pairs as each pair is expired.
        stop_pairs = iter(stop_pairs)

        # Stop labels of this EV, to be converted to pd.DataFrame and appended
        # to `all_stop_labels`.
        stop_labels_ev: List[Tuple[str, bool]] = []

        # Get the first stop entry-exit pair.
        (stop_entry, stop_exit) = next(stop_pairs)

        # Flag to keep track if vehicle is stopped or not.
        _stopped = False

        # Flag to indicate that all stops-pairs have been exhausted.
        _stop_pairs_exhausted = False

        for _, datapoint in trace_df.iterrows():
            # If the datapoint is stop_entry:
            if not _stop_pairs_exhausted:
                if datapoint['GPSID'] == stop_entry['GPSID']:
                    # set `_stopped`.
                    _stopped = True

                # If the datapoint is stop_exit:
                if datapoint['GPSID'] == stop_exit['GPSID']:
                    # reset `_stopped`. and try to get get next stop-pair.
                    _stopped = False
                    try:
                        # Get the next stop pair.
                        (stop_entry, stop_exit) = next(stop_pairs)

                        # If the new stop_entry is the current datapoint, set
                        # `_stopped`.
                        if datapoint['GPSID'] == stop_entry['GPSID']:
                            _stopped = True

                    # If there are no more stop pairs:
                    except StopIteration:
                        (stop_entry, stop_exit) = (None, None)
                        _stop_pairs_exhausted = True

            # Append the the status of `_stopped` to the labels of this EV.
            stop_labels_ev.append((datapoint['Time'], _stopped))

        stop_labels_ev_df = pd.DataFrame(stop_labels_ev, columns=['Time',
                                                                  'Stopped'])
        # Convert DateTime strings into proper pandas.DateTimes.
        stop_labels_ev_df['Time'] = pd.to_datetime(stop_labels_ev_df['Time'])

        # Append this EV's stop-labels to the list of all EVs.
        all_stop_labels.append((ev_name, stop_labels_ev_df))

    return all_stop_labels


def main(scenario_dir: Path):
    stops_output_file = scenario_dir.joinpath(
        'Temporal_Clusters', 'Clustered_Data',
        'dirty__time_clusters_with_dates.csv')

    # Calculate the stop arrival-times and the stop-durations for each EV:

    _ = input("Generate stop-arrivals and -durations? [y]/n ")
    if _.lower() != 'n':

        trace_dfs_generator = _gen_trace_dfs(scenario_dir)

        # Get the stop information.
        stops_df_filtered = \
            _build_stops_df(trace_dfs_generator)

        # Save the filtered stops dataframe.
        stops_df_filtered.to_csv(stops_output_file)

    # Create stop labels for each time-stamp:

    _ = input("Generate stop-labels (a label which specifies whether the " +
              "EV was stopped or not at a given timestamp)? [y]/n ")
    if _.lower() != 'n':

        trace_dfs_generator = _gen_trace_dfs(scenario_dir)

        # Get the stop labels.
        stop_labels_list = _build_stop_labels(trace_dfs_generator)

        # Save the stop labels.
        for (ev_name, stop_labels_df) in stop_labels_list:
            stop_labels_file = scenario_dir.joinpath(
                'Temporal_Clusters', 'Stop_Labels',
                f'stop_labels_{ev_name}.csv')
            stop_labels_df.to_csv(stop_labels_file, index=False)


if __name__ == "__main__":
    scenario_dir = Path(os.path.abspath(__file__)).parents[1]
    main(scenario_dir)
