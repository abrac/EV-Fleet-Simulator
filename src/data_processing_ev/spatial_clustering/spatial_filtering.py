#!/usr/bin/env python3

"""
Filters gps data-set to a geographical area.

Datapoints outside of the area are identified. Days in which these datapoints
are discorvered, are discarded from the dataset. Weekends are (optionally)
discarded.

Inputs:
    Taxi trace data for an extended time period (CSV).
        Preferably it should already have gone throught the clustering script.
    Geographical boundary (Geojson)
Outputs:
    Filtered gps datapoints (CSV)
    Statistics (txt) of:  # TODO
        Dates removed
        Percentage of datapoints found outside of boundary

Author: Chris Abraham
Date: 2020-08
"""

# %% Imports ##################################################################
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import datetime as dt
import data_processing_ev as dpr
import itertools
from multiprocessing import Pool, RLock, cpu_count


# %% Functions ################################################################
def is_point_in_poly(x: int, y: int, poly) -> bool:
    """
    Determine if the point is in the path.

    The algorithm uses the ["Even-Odd Rule"](https://en.wikipedia.org/wiki/
    Even–odd_rule)

    Args:
      x -- The x coordinates of point.
      y -- The y coordinates of point.
      poly -- a list of tuples [(x, y), (x, y), ...]

    Returns:
      True if the point is in the path.
    """
    num = len(poly)
    i = 0
    j = num - 1
    c = False
    for i in range(num):
        if ((poly[i][1] > y) != (poly[j][1] > y)) and \
                (x < poly[i][0] + (poly[j][0] - poly[i][0])
                                  * (y - poly[i][1])  # noqa
                                  / (poly[j][1] - poly[i][1])):  # noqa
            c = not c
        j = i
    return c


###############################################################################
def generate_bad_dates(clustered_trace: pd.DataFrame, boundary: pd.DataFrame,
                       ev_name: str, pid: int,
                       incl_weekends: bool = False, **kwargs) -> str:
    # TODO Check for public holidays as well!
    boundary = [tuple(vertex) for vertex in boundary.to_numpy()]
    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])

    prev_bad_date = None
    for _, datapoint in tqdm(clustered_trace.iterrows(),
                             desc=f"Generating bad dates {ev_name}",
                             total=len(clustered_trace), position=pid + 1):
        date = datapoint['Time'].split()[0]  # Time is "<date> <time>"
        # If this date is the same as the previous bad date, skip processing.
        if date == prev_bad_date:
            continue
        # If we're excluding weekends, check if date is a weekend
        is_bad_date = False
        if (not incl_weekends) and (
                input_data_fmt is not dpr.DATA_FMTS['GTFS']):
            day_of_week = dt.datetime(
                *[int(datepart) for datepart in date.split('-')]).weekday()
            # FIXME: Don't put '/' in this date.split. It's not a universal
            # solution for parsing dates.
            if day_of_week >= 5:
                is_bad_date = True
        if (is_bad_date or
                not is_point_in_poly(datapoint['Longitude'],
                                     datapoint['Latitude'], boundary)):
            prev_bad_date = date
            yield date
        else:
            continue


###############################################################################
def filter_cluster(cluster_file: Path, pid: int, boundary_file: Path,
                   output_path: Path, **kwargs):
    ev_name = cluster_file.stem.replace(' ', '_')  # e.g. T1000
    clustered_trace = pd.read_csv(cluster_file)
    # Remove outliers from data.
    clustered_trace = clustered_trace[clustered_trace['Cluster'] != -1]
    boundary = pd.read_csv(boundary_file, skipinitialspace=True,
                           sep='\\s*,\\s*', engine='python')
    bad_dates = [*generate_bad_dates(clustered_trace, boundary, ev_name, pid,
                                     **kwargs)]

    output_list = []
    prev_date = None
    for _, datapoint in tqdm(clustered_trace.iterrows(),
                             desc=f"Creating filtered traces {ev_name}",
                             total=len(clustered_trace), position=pid + 1):
        date = datapoint['Time'].split()[0]  # Time is "<date> <time>"
        # Output the dataframe as a file when a new date is encountered
        #   and clear the dataframe.
        if (date != prev_date and not output_list == []):
            # Create folder for this taxi if it doesn't exist
            output_subpath = output_path.joinpath(f"{ev_name}")
            output_subpath.mkdir(parents=True, exist_ok=True)
            output_df = pd.DataFrame(output_list)
            output_df.to_csv(
                output_subpath.joinpath(
                    "{0}_{1}.csv".format(ev_name,
                                         # change yy/mm/dd to yy-mm-dd
                                         prev_date.replace('/', '-'))),
                index=False)
            del output_list[:]  # clear the list
        if (date not in bad_dates):
            output_list.append(datapoint)
        prev_date = date

    # If there are still data remaining that hasn't been written:
    if not output_list == []:
        # Create folder for this taxi if it doesn't exist
        output_subpath = output_path.joinpath(f"{ev_name}")
        output_subpath.mkdir(parents=True, exist_ok=True)
        output_df = pd.DataFrame(output_list)
        output_df.to_csv(
            output_subpath.joinpath(
                "{0}_{1}.csv".format(ev_name,
                                     # change yy/mm/dd to yy-mm-dd
                                     prev_date.replace('/', '-'))),
            index=False)
        del output_list[:]  # clear the list


# %% Main #####################################################################
def filter_scenario(scenario_dir: Path, **kwargs):
    """
    Take clustered traces and filter out clusters outside of map boundary.

    Export the resulting dataframe as a csv file.
    """
    # TODO Make the below inputs function arguments.
    # TODO Make option of discarding weekends a command-line argument.
    clustered_files = sorted([*scenario_dir.joinpath(
        'Spatial_Clusters', 'Clustered_Traces').glob('*.csv')])
    boundary_file = sorted([*scenario_dir.joinpath('_Inputs', 'Map',
        'Boundary').glob('*.csv')])[0]
    output_path = scenario_dir.joinpath('Spatial_Clusters', 'Filtered_Traces')
    # Check if files exist in output_path.
    if any(output_path.glob('*/*.csv')):
        print(f"Warning: Files exist in {output_path}.\n\t" +
              "You may want to delete them!")
        dpr.auto_input("Press any key to continue...", '', **kwargs)
    num_jobs = len(clustered_files)

    # IF DEBUGGING:
    # -------------
    for idx, clustered_file in enumerate(clustered_files):
        filter_cluster(clustered_file, idx, boundary_file, output_path,
                       **kwargs)

    # IF NOT DEBUGGING:
    # -----------------
    # with Pool(processes=cpu_count()-1, initargs=(RLock(),),
    #           initializer=tqdm.set_lock) as p:
    #     args = zip(clustered_files, range(num_jobs),
    #                itertools.repeat(boundary_file, num_jobs),
    #                itertools.repeat(output_path, num_jobs),
    #                itertools.repeat(**kwargs, num_jobs))
    #     p.starmap(filter_cluster, args)

    # Print these blanks so that tqdm bars can stay
    print("\n" * (num_jobs + 0))
