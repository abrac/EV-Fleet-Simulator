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

import os
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Iterator
import datetime as dt
from itertools import repeat
from tqdm import tqdm
from haversine import haversine
import data_processing_ev as dpr


def _reformat_time(time_str: str, **kwargs):
    """Takes a date-time string, and returns either a dt.datetime or
       dt.timedelta object for 'GPS' and 'GTFS' data formats respectively."""
    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])
    if input_data_fmt == dpr.DATA_FMTS['GPS']:
        date, time = time_str.split()
        hr24, minute, sec = [int(time_old) for time_old in time.split(':')]
        date_arr = [int(date_part) for date_part in date.split('-')]
        formatted_time = dt.datetime(*date_arr, hr24, minute, sec)
    elif input_data_fmt == dpr.DATA_FMTS['GTFS']:
        day_d, time = time_str.split()
        hr24, minute, sec = [int(time_old) for time_old in time.split(':')]
        days = int(day_d.split('d')[0])
        formatted_time = dt.timedelta(
            days=days, hours=hr24, minutes=minute, seconds=sec)
    else:
        raise ValueError(dpr.DATA_FMT_ERROR_MSG)
    return formatted_time

    # date, time, meridiem = datetime_str.split()
    # hr12, minute, sec = [int(time_old) for time_old in
    #                      time.split(':')]
    # hr24 = (hr12 % 12) + (12 if meridiem == "PM" else 0)
    # date_arr = [int(date_part) for date_part in date.split('/')]
    # return dt.datetime(*date_arr, hr24, minute, sec)


def reformat_time2timetuple_str(hms_str: str) -> str:
    hour, minute, second = (int(x) for x in hms_str.split(':'))
    timedelta = dt.timedelta(hours=hour, minutes=minute, seconds=second)
    days = timedelta.days
    hms = str(timedelta).split(', ')[-1]
    hour, minute, second = hms.split(':')
    new_hour = int(hour)  # * 24
    new_time = f'{days}d {new_hour:02d}:{minute}:{second}'

    return new_time


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


def _get_stop_entries_and_exits(trace_df: pd.DataFrame, ev_name: str, **kwargs
                                ) -> List[Tuple]:
    """
    Returns a list of tuples representing the entries and exits of each of
    the stops.

    If the input data format is 'GPS', the tuple will consist of two pd.Series
    objects. The first corresponding to the stop entry datapoint, and the second
    corresponding to the stop exit datapoint.

    If the input data format is 'GTFS', the tuple will consist of two string
    objects, and one pd.Series object. The first string corresponds to the day
    and time that the vehicle arrived at the stop, the second corresponds to
    the day and time that the vehicle departed from the stop, and pd.Series
    object corresponds to datapoint of the stop from the trace_df.
    """

    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])

    stop_entries_and_exits = []

    # These are the edges cases that we need to consider:
    #   1. [x] Taxi is stopped at the first datapoint. Create stop event from
    #          0h until the first *non-stopped* datapoint. Record the location
    #          of the first datapoint.
    #   2. [x] Taxi is running at the first datapoint. Create stop event from
    #          0h until first datapoint. Use location of first datapont.
    #   3. [x] Taxi is stopped at the last datapoint. Create stop event from
    #          last datapoint until midnight. Use location of last datapoint.
    #   4. [x] Taxi is running at the last datapoint. Do not create a stop
    #          event.

    # GPS version.
    if input_data_fmt == dpr.DATA_FMTS['GPS']:
        prev_datapoint = None  # Bool
        entry_datapoint = None  # First datapoint in the stop event.
        start_new_entry = None  # Bool
        stop_encountered = None  # Bool
        for _, datapoint in trace_df.iterrows():

            curr_date = datapoint['Time'].split(' ')[0]

            # In the first iteration, we are going to create a synthetic
            # datapoint at 00h00, with the real first datapoint's location. We
            # will set that datapoint as the `entry_datapoint`.
            if prev_datapoint is None:
                # If this is the first iteration of the whole dataframe:
                first_iteration_of_date = True
            elif curr_date != prev_datapoint['Time'].split(' ')[0]:
                # If the current date does not match the previous date:
                first_iteration_of_date = True
            else:
                # Else (i.e. The previous date is the same as the current
                # date):
                first_iteration_of_date = False

            if first_iteration_of_date:

                # Create a final stop event until midnight, for the previous
                # date. If there is already an unconcluded stop event from the
                # previous date, conclude it at midnight of that date. Else,
                # create a stop event which starts at the final datapoint of
                # the previous date, and ends at midnight of that date.
                if prev_datapoint is not None:
                    if stop_encountered:
                        # Make a copy of the final datapoint from the previous
                        # date, and change its time to midnight.
                        exit_datapoint = prev_datapoint.copy()
                        exit_datapoint['GPSID'] = -exit_datapoint['GPSID']
                        exit_datapoint['Time'] = (exit_datapoint['Time'].\
                                                  split(' ')[0] + " 23:59:59")
                        exit_datapoint['Velocity'] = 0
                        # TODO TODO DELETE DATES WHICH ONLY HAVE ONE STOP EVENT.
                        # if entry_datapoint['GPSID'] < 0:
                        #     # Discard the stop-event if the taxi didn't move at
                        #     # all on the previous date.
                        #     pass
                        # else:
                        #     stop_entries_and_exits.append((entry_datapoint,
                        #                                    exit_datapoint))
                        stop_entries_and_exits.append((entry_datapoint,
                                                       exit_datapoint))
                    else:
                        entry_datapoint = prev_datapoint.copy()
                        entry_datapoint['Velocity'] = 0
                        exit_datapoint = prev_datapoint.copy()
                        exit_datapoint['GPSID'] = -exit_datapoint['GPSID']
                        exit_datapoint['Time'] = (exit_datapoint['Time'].\
                                                  split(' ')[0] + " 23:59:59")
                        exit_datapoint['Velocity'] = 0
                        stop_entries_and_exits.append((entry_datapoint,
                                                       exit_datapoint))

                    stop_encountered = False

                # Create a synthetic datapoint with the first datapoint's
                # location, but the time set at 00h00
                entry_datapoint = datapoint.copy()
                entry_datapoint['GPSID'] = -entry_datapoint['GPSID']
                entry_datapoint['Time'] = (datapoint['Time'].split(' ')[0] +
                                           " 00:00:00")
                entry_datapoint['Velocity'] = 0

                stop_encountered = True
                first_iteration_of_date = False


            # Check for consecutive points that have at least one point where
            # ... the taxi was stopped.
            # Continue until the taxi moved at a speed greater than or equal to
            # ... 10 km/h or the taxi drifts 25m from the stop location.
            if stop_encountered:
                current_location = (datapoint['Latitude'],
                                    datapoint['Longitude'])
                stop_location = (entry_datapoint['Latitude'],
                                 entry_datapoint['Longitude'])
                distance_drifted = haversine(current_location,
                                             stop_location, unit='m')
                if datapoint['Velocity'] >= 10 or distance_drifted >= 25:
                    start_new_entry = True

            # If the taxi stopped stopping:
            if start_new_entry:
                # Record the entry_datapoint and the current datapoint. We are
                # considering the current datapoint's timestamp to be the *end*
                # of the stop-event -- otherwise, single-datapoint stop-events
                # will be lost...
                stop_entries_and_exits.append(
                    (entry_datapoint, datapoint))
                # Reset the flags
                start_new_entry = False
                stop_encountered = False

            # Look for a new stop-entry
            if datapoint['Velocity'] < 1 and not stop_encountered:
                # if this is the first stop that was encountered, make it the
                # "entry" datapoint of the stop event and record its location.
                entry_datapoint = datapoint
                stop_encountered = True

            prev_datapoint = datapoint

        # If, after the loop, there is a stop_entry remaining without a
        # stop_exit:
        if stop_encountered:
            # Make a copy of the final datapoint, and change its time to
            # midnight.
            exit_datapoint = datapoint.copy()
            exit_datapoint['GPSID'] = -exit_datapoint['GPSID']
            exit_datapoint['Time'] = (exit_datapoint['Time'].split(' ')[0] +
                                      " 23:59:59")
            exit_datapoint['Velocity'] = 0
            stop_entries_and_exits.append((entry_datapoint, exit_datapoint))
            stop_encountered = False

    # GTFS version.
    # TODO: Account for GTFS data which do not specify the arrival time
    #       separate from the departure_time.
    elif input_data_fmt == dpr.DATA_FMTS['GTFS']:

        stop_times_df = kwargs.get('GTFS_stop_times_df', None)

        stop_times_df = stop_times_df[stop_times_df['trip_id'] == ev_name]

        # For each row of trace_df, check if the row has a value for `stop_id`.
        # If it does, it is a stop. Find the stop entry and exit time from
        # stop_times_df.

        stops = stop_times_df.iterrows()  # An iterator which keeps track
            # of the next anticipated stop. The generator will be triggered
            # after each stop is found in the traces_df.
        _, next_stop = next(stops)
        for _, datapoint in trace_df.iterrows():
            # Check for a point that is defined as a stop. I.e., it should have
            # a value for it's `stop_id`.
            if not pd.isna(datapoint['StopID']):
                # Retrieve the stop-arrival and -departure times.
                stop_id = datapoint['StopID']

                # But first do a bit of error checking.
                while stop_id != next_stop['stop_id']:
                    # TODO Use logging.warn() instead of print().
                    input("\n\nWarning: A stop in the sequence in " +
                          "stop_times.txt is missing from the gps trace. " +
                          "Press enter to skip this stop, and check the if "
                          "the next stop in the sequence is in the gps trace.")
                    try:
                        _, next_stop = next(stops)
                    # If there are no more stop pairs:
                    except StopIteration:
                        break


                arrival = reformat_time2timetuple_str(
                    next_stop['arrival_time'])
                departure = reformat_time2timetuple_str(
                    next_stop['departure_time'])

                # Record a tuple of: `entry_time`, `departure_time` and
                # `stop_id`.
                stop_entries_and_exits.append((arrival, departure, datapoint))

                try:
                    _, next_stop = next(stops)
                # If there are no more stop pairs:
                except StopIteration:
                    continue
    else:
        raise ValueError(dpr.DATA_FMT_ERROR_MSG)

    return stop_entries_and_exits


def _build_stops_df(trace_dfs_generator: Iterator[Tuple[pd.DataFrame, str]],
                    **kwargs) -> Tuple[pd.DataFrame,
                                       List[Tuple[str, pd.Series, pd.Series]]]:
    """Calculate the stops in the GPS traces.

    Returns:
        Tuple of:
            - Dataframe of stop-arrivals and -durations.
            - List of stop entries and exits.
    """

    def _filter_stop_events(stops_df: pd.DataFrame, dropping=True,
                            duration_limits: Tuple[float, float] = [0.33, 24],
                            arrival_limits: Tuple[float, float] = [0, 24]):
        min_stop_duration = duration_limits[0]
        max_stop_duration = duration_limits[1]
        min_stop_arrival = arrival_limits[0]
        max_stop_arrival = arrival_limits[1]
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

    stops_dfs_filtered = []
    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])

    # Generate the dataframe of stop-arrivals and -durations for each EV, and
    # append to trace_dfs.
    for trace_df, trip_name in tqdm(trace_dfs_generator):

        # First: Generate a list of stop-entry and exit times.
        stop_entries_and_exits = _get_stop_entries_and_exits(
            trace_df, trip_name, **kwargs)

        # If list of stop_times still empty, throw exception, and continue to
        # next EV.
        if stop_entries_and_exits == []:
            print(f"Error: no stops found for EV {trip_name}.")
            continue

        # Secondly: Calulate stop-durations from stop-entry and -exit times.
        if input_data_fmt == dpr.DATA_FMTS['GPS']:
            stop_durations = []
            for (stop_entry, stop_exit) in stop_entries_and_exits:
                entry_time = _reformat_time(stop_entry['Time'], **kwargs)
                exit_time = _reformat_time(stop_exit['Time'], **kwargs)
                stop_duration = (exit_time - entry_time).total_seconds() / 3600
                if stop_duration < 0:
                    raise ValueError("Stop duration was negative!")
                stop_durations.append(stop_duration)
            # Convert stop-arrival times from date-time objects to floats which
            # represent the hour arrived.
            stop_arrivals = [
                dt.timedelta(hours=entry_time.hour, minutes=entry_time.minute,
                             seconds=entry_time.second).total_seconds() / 3600
                for entry_time in [_reformat_time(stop_entry['Time'], **kwargs)
                                   for (stop_entry, _) in
                                   stop_entries_and_exits]]
            dates = [
                entry_datetime.date() for
                entry_datetime in [_reformat_time(stop_entry['Time'], **kwargs)
                                   for (stop_entry, _) in
                                   stop_entries_and_exits]]
            cluster_nums = [stop_entry['Cluster'] for (stop_entry, _) in
                            stop_entries_and_exits]

            stops_df = pd.DataFrame(
                zip(
                    dates, stop_arrivals, stop_durations, cluster_nums
                ),
                columns=['Date', 'Stop_Arrival', 'Stop_Duration', 'Cluster']
            )

            # Thirdly: Remove stop-events that are irrelevant.
            stops_df = _filter_stop_events(stops_df, dropping=True)
            # If all the stops were filtered out, throw warning, and continue.
            if stops_df.empty:
                print(f"Warning: all stops filtered out for EV {trip_name}.")
                    # TODO Change to logging.warn # noqa
                continue

            # Add "ev_name" as a columns to the dataframe, and then set
            # "ev_name" & "date" as indices.
            stops_df['EV_Name'] = trip_name
            stops_df = stops_df.set_index(['EV_Name', 'Date'])

            # Finally: Append the data frame and the ev_name to the lists.
            stops_dfs_filtered.append(stops_df)

        elif input_data_fmt == dpr.DATA_FMTS['GTFS']:
            stop_durations = []
            for (stop_entry_str, stop_exit_str, _) in \
                    stop_entries_and_exits:
                entry_time = _reformat_time(stop_entry_str, **kwargs)
                exit_time = _reformat_time(stop_exit_str, **kwargs)
                stop_duration = (exit_time.total_seconds() -
                                 entry_time.total_seconds()) / 3600
                if stop_duration < 0:
                    raise ValueError("Stop duration was negative!")
                stop_durations.append(stop_duration)
            # Convert stop-arrival times from timedelta objects to floats which
            # represent the hour arrived.
            stop_arrivals = [
                entry_time.seconds / 3600
                for entry_time in [_reformat_time(stop_entry_str, **kwargs) for
                                   (stop_entry_str, _, _) in
                                   stop_entries_and_exits]]
            days = [
                entry_time.days for
                entry_time in [_reformat_time(stop_entry_str, **kwargs) for
                               (stop_entry_str, _, _) in
                               stop_entries_and_exits]]
            cluster_nums = [datapoint['Cluster'] for (_, _, datapoint) in
                            stop_entries_and_exits]

            stops_df = pd.DataFrame(
                zip(
                    days, stop_arrivals, stop_durations, cluster_nums
                ),
                columns=['Date', 'Stop_Arrival',
                         'Stop_Duration', 'Cluster']
            )

            # Thirdly: Remove stop-events that are irrelevant.
            stops_df = _filter_stop_events(stops_df, dropping=True,
                                           duration_limits=[0, 8])
                # TODO: Check if 0 is an appopriate limit. I have only set this
                # for the GTFS data format because very few or no long stops
                # are defined. I will need to take this change into account
                # when calculating the charging potential in the results
                # analysis.

            # If all the stops were filtered out, throw warning, and continue.
            if stops_df.empty:
                print(f"Warning: all stops filtered out for EV {trip_name}.")
                    # TODO Change to logging.warn # noqa
                continue

            # Add "ev_name" as a columns to the dataframe, and then set "ev_name" &
                # "date" as indices.
            stops_df['EV_Name'] = trip_name
            stops_df = stops_df.set_index(['EV_Name', 'Date'])

            # Finally: Append the data frame and the ev_name to the lists.
            stops_dfs_filtered.append(stops_df)

        # If the input data format was not GPS or GTFS raise an error.
        else:
            raise ValueError(dpr.DATA_FMT_ERROR_MSG)

    # Combine the dataframes in the lists into one.
    full_stops_df_filtered = pd.concat(stops_dfs_filtered)

    return full_stops_df_filtered


def _build_stop_labels(
            trace_dfs_generator: Iterator[Tuple[pd.DataFrame, str]], **kwargs
        ) -> List[Tuple[str, pd.DataFrame]]:
    """
    Build stop labels corresponding to each datpoint in each EV.
    """

    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])

    all_stop_labels: List[pd.DataFrame] = []  # The stop_labels of each EV will
        # be appended onto this list.

    if input_data_fmt == dpr.DATA_FMTS['GPS']:
        for (trace_df, trip_name) in tqdm(trace_dfs_generator):

            # Get the pairs of entries and exits for this trace.
            stop_pairs = _get_stop_entries_and_exits(trace_df, trip_name, **kwargs)

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
                    # If the datapoint is the stop-entry, or it is the first
                    # datapoint of the day.
                    if (datapoint['GPSID'] == stop_entry['GPSID'] or
                            datapoint['GPSID'] == -stop_entry['GPSID']):
                        # set `_stopped`.
                        _stopped = True

                    # If the datapoint is the stop_exit, or it is the last
                    # datapoint of the day.
                    if (datapoint['GPSID'] == stop_exit['GPSID'] or
                            datapoint['GPSID'] == -stop_exit['GPSID']):
                        # If the datapoint is the stop_exit, reset `_stopped`.
                        if datapoint['GPSID'] == stop_exit['GPSID']:
                            _stopped = False
                        # If the datapoint is the last datapoint of the day,
                        # set `_stopped`.
                        if datapoint['GPSID'] == -stop_exit['GPSID']:
                            _stopped = True
                        # Try to get get next stop-pair.
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
            all_stop_labels.append((trip_name, stop_labels_ev_df))

    elif input_data_fmt == dpr.DATA_FMTS['GTFS']:

        for (trace_df, trip_name) in tqdm(trace_dfs_generator):
            stop_labels_df = pd.DataFrame()
            stop_labels_df['Time'] = trace_df['Time']
            stop_labels_df['Stopped'] = pd.notna(trace_df['StopID'])
            all_stop_labels.append((trip_name, stop_labels_df))

    else:
        raise ValueError(dpr.DATA_FMT_ERROR_MSG)

    return all_stop_labels


def extract_stops(scenario_dir: Path, **kwargs):
    stops_output_file = scenario_dir.joinpath(
        'Temporal_Clusters', 'Clustered_Data',
        'time_clusters_with_dates.csv')

    kwargs['scenario_dir'] = scenario_dir
    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])

    if input_data_fmt == dpr.DATA_FMTS['GTFS']:
        stop_times_df = pd.read_csv(
            scenario_dir.joinpath(
                '_Inputs', 'Traces', 'Original', 'GTFS', 'stop_times.txt'),
            dtype={'trip_id': str},
            # dtype={'trip_id': str, 'arrival_time': str, 'departure_time': str,
            #        'stop_id': str, 'stop_sequence': int, 'stop_headsign': str,
            #        'pickup_type': int, 'drop_off_type': int,
            #        'continuous_pickup': int, 'continuous_drop_off': int,
            #        'shape_dist_traveled': float, 'timepoint': int},
            skipinitialspace=True)

        kwargs['GTFS_stop_times_df'] = stop_times_df

    # Calculate the stop arrival-times and the stop-durations for each EV:

    _ = input("Generate stop-arrivals and -durations? [y]/n ")
    if _.lower() != 'n':

        trace_dfs_generator = _gen_trace_dfs(scenario_dir)

        # Get the stop information.
        stops_df_filtered = \
            _build_stops_df(trace_dfs_generator, **kwargs)

        # Save the filtered stops dataframe.
        stops_df_filtered.to_csv(stops_output_file)

    # Create stop labels for each time-stamp:

    _ = input("Generate stop-labels (a label which specifies whether the " +
              "EV was stopped or not at a given timestamp)? [y]/n ")
    if _.lower() != 'n':

        trace_dfs_generator = _gen_trace_dfs(scenario_dir)

        # Get the stop labels.
        stop_labels_list = _build_stop_labels(trace_dfs_generator, **kwargs)

        # Save the stop labels.
        for (trip_name, stop_labels_df) in stop_labels_list:
            stop_labels_file = scenario_dir.joinpath(
                'Temporal_Clusters', 'Stop_Labels',
                f'stop_labels_{trip_name}.csv')
            stop_labels_df.to_csv(stop_labels_file, index=False)
