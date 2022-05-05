from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from scipy.stats.kde import gaussian_kde
import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat
import pickle
import logging
from haversine import haversine  # Calculates distance between geo coordinates.

INPUT_TYPES = {'clustered': 'spatial_clustered_traces',
               'filtered': 'spatial_filtered_traces'}
logging.basicConfig(level=logging.INFO)


def _time_str2dt_old(datetime_str: str) -> dt.time:
    """
    Delete this function. It was specifically for the Stellenbosch data's
    format. I.e: The data had AM and PM in it. The new way, is to make sure
    that pre-processing pre-formats the datetimes into the
    "YYYY-MM-DD HH:MM:SS" format.
    """
    date, time, meridiem = datetime_str.split()
    hr12, minute, sec = [int(time_old) for time_old in
                         time.split(':')]
    hr24 = (hr12 % 12) + (12 if meridiem == "PM" else 0)
    date_arr = [int(date_part) for date_part in date.split('/')]
    return dt.datetime(*date_arr, hr24, minute, sec)


def _time_str2dt(datetime_str: str) -> dt.datetime:
    date, time = datetime_str.split()
    hr24, minute, sec = [int(time_old) for time_old in
                         time.split(':')]
    date_arr = [int(date_part) for date_part in date.split('-')]
    return dt.datetime(*date_arr, hr24, minute, sec)

def _gen_trace_dfs(scenario_dir: Path, input_type: str
                   ) -> Iterator[Tuple[pd.DataFrame, str]]:
    # Check if program must read traces from the "clustered-only" dataset
    # or "clustered-and-filtered" dataset.
    # TODO: Ignore the "clustered-only" option. It's outdated and should be
    #       removed.
    if input_type == INPUT_TYPES['clustered']:
        # TODO Merge changes from gen_trace_dfs() in 'filtered' input type
        trace_files = scenario_dir.joinpath(
            'Spatial_Clusters', 'Clustered_Traces').glob('*.csv')
        for trace_file in trace_files:
            trace_df = pd.read_csv(trace_file)
            yield (trace_df, trace_file.stem)
    elif input_type == INPUT_TYPES['filtered']:
        # Get the location of the filtered traces.
        traces_root_dir = scenario_dir.joinpath('Spatial_Clusters',
                                                'Filtered_Traces')
        # create an array of their directories, corresponding to each EV.
        trace_dirs = sorted(
            [f for f in traces_root_dir.iterdir() if f.is_dir()])
        # If there are no traces, throw an error.
        if len(trace_dirs) < 1:
            raise ValueError('No traces found in \n\t{0}.'.format(
                traces_root_dir.absolute()))
        # For each EV:
        for trace_dir in trace_dirs:
            # Get the trace files and sort them by date.
            trace_files = sorted(trace_dir.glob('T*.csv'))
            # Convert the traces to dataframes and append them to a list.
            trace_dfs = []
            for trace_file in trace_files:
                trace_dfs.append(pd.read_csv(trace_file))
            # Convert the list of dataframes to huge dataframe.
            trace_df = pd.concat(trace_dfs, ignore_index=True)
            # Yield it.
            yield (trace_df, trace_dir.stem)
    else:
        raise ValueError('Input type is neither one of: \n\t',
                         INPUT_TYPES.values())


def _gen_stop_pdfs(trace_df: pd.DataFrame, ev_name: str,
                   **kwargs) -> Iterator[Tuple[int, np.array, plt.Figure]]:
    """Generate a probability density curve from the given trace inputs."""
    # For each spatial_cluster, create a list of stop times.
    #   List of unique clusters
    auto_run = kwargs.get('auto_run', False)
    clusters = trace_df['Cluster'].unique()
    clusters.sort()
    for cluster in clusters:
        # Create dataframe of cluster
        trace_df_cluster = trace_df[trace_df['Cluster'] == cluster]
        # Take only rows where velocity < 1
        trace_df_cluster = trace_df_cluster[trace_df['Velocity'] < 1]
        # Generate a list of stop times
        stop_times = []
        for _, datapoint in trace_df_cluster.iterrows():
            time = _time_str2dt(datapoint['Time'])
            stop_times.append(time)
        # If list of stop_times still empty, throw exception, and continue
        if stop_times == []:
            logging.warn(f"No stop times found for cluster {cluster}.")
            if not auto_run:
                input("Press any key to continue...")
            continue
        stop_time_floats = np.array([
            dt.timedelta(hours=t.hour, minutes=t.minute,
                         seconds=t.second).total_seconds()/3600 for
            t in stop_times])
        # Create temporal_clusters from the list of stop times.
        t_domain = np.linspace(0, 24, 1000)
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': np.linspace(0, 1.0, 200)})
        try:
            grid.fit(stop_time_floats[:, None])
        except ValueError as e:
            logging.error(e)
            if not auto_run:
                input('Press any key to acknowledge...')
            continue
        logging.info(f"Best bandwidth: {grid.best_params_}")
        kde = grid.best_estimator_
        pdf = np.exp(kde.score_samples(t_domain[:, None]))

        # Plot results:
        fig, ax = plt.subplots()
        ax.plot(t_domain, pdf, linewidth=3, alpha=0.5,
                label='bw=%.2f' % kde.bandwidth)
        points = ax.plot(stop_time_floats, [0 for _ in stop_time_floats],
                         marker='x', linestyle='None',
                         label='Recorded stop times.')[0]
        points.set_clip_on(False)
        ax.hist(stop_time_floats, 24, fc='gray',
                histtype='stepfilled', alpha=0.3, density=True)
        ax.legend()
        ax.set_title(f"Probability density curve of taxi '{ev_name}' "
                     f"being stopped at cluster {cluster}*")
        fig.text(0, 0, '*Assuming taxi stops at the cluster on a given ' +
                 'day.')
        ax.set_xlabel('Time of Day (Hour)')
        ax.set_ylabel('Probability Density')
        if not auto_run:
            input("Press any key to continue...")
        yield (cluster, pdf, fig)


def _gen_stop_duration_pdfs(scenario_dir: Path, trace_df: pd.DataFrame,
                            ev_name: str
                            ) -> Iterator[Tuple[int, np.array, plt.Figure]]:
    # For each spatial_cluster, create a list of arrival_times and
    # stop_durations.
    clusters = trace_df['Cluster'].unique()
    clusters.sort()
    for cluster in clusters:
        # Code Reuse ↓ # TODO  # TODO Delete this todo...
        # Create dataframe of cluster
        trace_df_cluster = trace_df[trace_df['Cluster'] == cluster]

        # Generate a list of stop-arrival times and stop durations.
        # ---------------------------------------------------------

        # First: Generate a list of stop-entry and exit times.
        stop_entries_and_exits: List[Tuple[pd.Series, int]] = []
        prev_datapoint = None
        entry_datapoint = None
        start_new_entry = False
        stop_encountered = False
        stop_location = None  # Coordinates of the first datapoint in the stop.
        for _, datapoint in trace_df_cluster.iterrows():  # TODO Should I really do stop-extraction on a per-cluster basis?
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

        # If list of stop_times still empty, throw exception, and continue to
            # next cluster.
        if stop_entries_and_exits == []:
            logging.error(f"No stops found for cluster {cluster}.")
            continue

        # Second: Calulate stop-durations from stop-entry and -exit times.
        stop_durations = []
        for (stop_entry, stop_exit) in stop_entries_and_exits:
            entry_time = _time_str2dt(stop_entry['Time'])
            exit_time = _time_str2dt(stop_exit['Time'])
            stop_duration = (exit_time-entry_time).total_seconds()/3600
            if stop_duration < 0:
                raise ValueError("Stop duration was negative!")
            stop_durations.append(stop_duration)
        # Convert stop-arrival times from date-time objects to a float,
            # which represents the hour arrived.
        stop_arrivals = [
            dt.timedelta(hours=entry_time.hour, minutes=entry_time.minute,
                         seconds=entry_time.second).total_seconds()/3600 for
            entry_time in [_time_str2dt(stop_entry['Time']) for
                           (stop_entry, _) in stop_entries_and_exits]]
        # Code Reuse ↑ # TODO  # TODO Delete this todo...

        # (Optics) Clustering/3D PDF of scatterplot
        #   Prepare data for model
        stops_df = pd.DataFrame(zip(stop_arrivals, stop_durations),
                                columns=['Stop_Arrival', 'Stop_Duration'])
        #   Normalise data  # FIXME Don't do this anymore.
        # stops_df_data_scaler = StandardScaler().fit(stops_df)
        # stops_scaled_arr = stops_df_data_scaler.transform(stops_df)
        stops_scaled_arr = stops_df
        #   TODO Find optimum min_samples and max_eps
        #   Construct the model
        min_samples = 5
        if len(stops_scaled_arr) >= min_samples:
            # FIXME Increase min_samples
            model = OPTICS(min_samples=min_samples, max_eps=1,
                           metric='euclidean', cluster_method='dbscan',
                           n_jobs=None  # too much overhead if multi-processing # noqa
                           ).fit(stops_scaled_arr)
            stops_df['Cluster'] = model.labels_
        else:
            model = None
            stops_df['Cluster'] = -1

        # Remove stop-events that are irrelevant.
        def _stops_mask(stops_df: pd.DataFrame, dropping=False):
            max_stop_duration = 8
            min_stop_duration = 0.33
            min_stop_arrival = 0
            max_stop_arrival = 24
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
        stops_df = _stops_mask(stops_df, dropping=True)

        # If all the stops were filtered out, throw warning, and continue.
        if stops_df.empty:
            logging.warn(f"All stops filtered out for cluster {cluster}.")
            continue

        # 2D Scatterplot of arrival_times and stop_durations
        unique_time_clusters = set(stops_df['Cluster'])
        fig, ax = plt.subplots(figsize=(3, 2.5))
        for time_cluster in unique_time_clusters:
            if time_cluster == -1:
                # White/black used for noise.
                kwargs = {'markersize': 4, 'color': 'black',
                          'label': 'Outliers', 'markeredgecolor': 'white',
                          'markeredgewidth': 0.5}
            else:
                # FIXME: Making color black for now. Remove if you want
                # different colours for each temporal-cluster.
                kwargs = {'markersize': 2, 'color': 'black',
                          'label': f'Time-cluster {time_cluster}'
                          # 'markeredgecolor': 'white', 'markeredgewidth': 0.5
                          # 'alpha': 0.5
                          }
            stops_cluster_df = stops_df[stops_df['Cluster'] == time_cluster]
            ax.plot(stops_cluster_df['Stop_Arrival'],
                    stops_cluster_df['Stop_Duration'], marker='.',
                    linestyle='None', **kwargs)
        # FIXME: Hiding legend for now.
        # ax.legend()
        # ax.set_title("Stop Arrivals and Durations at Spatial-Cluster " +
        #              f"{cluster} for '{ev_name}'")

        # Plot gaussian distribution
        nbins = 100

        x = np.array(stops_df['Stop_Arrival'])
        y = np.array(stops_df['Stop_Duration'])
        # ax.hist2d(x, y, bins=nbins, alpha=0.5)
        # plt.Axes.hist2d

        # x_min, x_max = x.min(), x.max()
        # y_min, y_max = y.min(), y.max()
        ### XXX Uncomment above and comment below. ###
        x_min, x_max = 5, 21
        y_min, y_max = 0, 4

        xi, yi = np.mgrid[0: 24: nbins * 1j,
                          0: y.max() + 4: nbins * 1j]

        # Plot box-plots
        stops_in_hours = []
        for hour in range(x_min, x_max):
            stops_in_hour = []
            for stop_arrival, stop_duration in zip(x, y):
                if int(stop_arrival) == hour:
                    stops_in_hour.append(stop_duration)
            stops_in_hours.append(stops_in_hour)
        bp0 = ax.boxplot(
            stops_in_hours,
            positions=[x + 0.5 for x in range(x_min, x_max)],
            showfliers=False,
            manage_ticks=False,
            medianprops={'color': 'black'}
            )
        bp1 = ax.boxplot(
            stops_in_hours,
            positions=[x + 0.5 for x in range(x_min, x_max)],
            showfliers=False,
            manage_ticks=False,
            medianprops={'color': 'black'},
            )
        # for component in ['whiskers', 'caps', 'medians', 'boxes']:
        #     plt.setp(bp0[component], path_effects=[
        #         PathEffects.withStroke(linewidth=3, foreground="w")
        #         # PathEffects.SimpleLineShadow(shadow_color='w'),
        #         # PathEffects.Normal()
        #         ])
        for component in ['whiskers', 'caps', 'medians', 'boxes']:
            plt.setp(bp0[component], color='white', linewidth=4)
            plt.setp(bp1[component], linewidth=1.5)
        plt.setp(bp0['medians'], zorder=2)

        # cmap = plt.get_cmap('Greys')
        # k = gaussian_kde(np.vstack([x, y]))
        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # ax.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5,
        #               shading='auto', cmap=cmap)
        # cs = ax.contour(xi, yi, zi.reshape(xi.shape), alpha=1, colors=['black'])
        # ax.clabel(cs, cs.levels, inline=True, manual=True)
        # cs = ax.contour(xi, yi, zi.reshape(xi.shape), alpha=1, cmap=cmap)
        # cs = ax.contourf(xi, yi, zi.reshape(xi.shape), alpha=1, cmap=cmap)
        # colorbar = fig.colorbar(cs, ax=ax, label='Probability Density')
        # colorbar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        # colorbar.ax.yaxis.values *= 100

        # ax.set_xlabel("Time of arrival (Hour of day)")
        # ax.set_ylabel("Duration of stop (Hours)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.set_xticks([6, 8, 10, 12, 14, 16, 18])  # XXX

        fig.tight_layout()

        yield (cluster, stops_df, fig)


def _cluster_ev(trace_df: pd.DataFrame, ev_name: str, scenario_dir: Path,
                **kwargs) -> pd.DataFrame:
    auto_run = kwargs.get('auto_run', False)
    # Individual time clustering:
    logging.info(f'### Starting time-clustering of {ev_name} ###')
    output_dir = scenario_dir.joinpath('Temporal_Clusters', 'Graphs',
                                       'Stop_PDFs', ev_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    # for cluster_num, pdf, fig in _gen_stop_pdfs(trace_df,
    #                                             ev_name, **kwargs):
    #     fig.show()
    #     output_file = output_dir.joinpath(f'{cluster_num}.svg')
    #     fig.savefig(output_file)

    # Record total stops at each time cluster for ea spatial
    # cluster as json (and graphs).
    # TODO

    # Create plot of arrival time and stop duration for ea spatial
    # cluster as well as 3D PDFs.
    graph_dir = scenario_dir.joinpath('Temporal_Clusters', 'Graphs',
                                      'Stop_Duration_PDFs', ev_name)
    save_dir = scenario_dir.joinpath('Temporal_Clusters',
                                     'Clustered_Data', ev_name)
    graph_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    clustered_dfs = []
    for cluster_num, clustered_df, fig in _gen_stop_duration_pdfs(
            scenario_dir, trace_df, ev_name):
        output_file = graph_dir.joinpath(f'{ev_name}_{cluster_num}.svg')
        fig.savefig(output_file)
        output_file = graph_dir.joinpath(f'{ev_name}_{cluster_num}.pdf')
        fig.savefig(output_file)
        output_file = graph_dir.joinpath(f'{ev_name}_{cluster_num}.png')
        fig.savefig(output_file, dpi=600)
        # Save interactive figure
        output_file = graph_dir.joinpath(f'{cluster_num}.fig.pickle')
        pickle.dump(fig, open(output_file, 'wb'))
        #   Save the dataframe
        clustered_df.to_csv(save_dir.joinpath(
            f'cluster_{cluster_num}.csv'), index=False)
        # Making the time_clusters the index of clustered_df
        clustered_df_gb = clustered_df.groupby('Cluster',
                                               group_keys=True)
        clustered_df = clustered_df_gb.apply(
            lambda x: x.sort_values(
                by='Stop_Duration', ascending=False)
        ).drop('Cluster', 1)
        clustered_df.index.names = ['Time_Cluster', 'Row_No']
        # Prepend space_cluster as index of clustered_df
        clustered_df = pd.concat({str(cluster_num): clustered_df},
                                 names=['Space_Cluster'])
        clustered_dfs.append(clustered_df)
        plt.close()
    try:
        vehicle_df = pd.concat(clustered_dfs)
        # Prepend vehicle_name as index of vehicle_df
        vehicle_df = pd.concat({str(ev_name): vehicle_df},
                               names=['Vehicle_Name'])
    except ValueError as e:
        logging.error(str(e) + f"This is for vehicle {ev_name}.")
        vehicle_df = None
        if not auto_run:
            input("Press enter to accept... ")
    logging.info(f'    *** Finished time-clustering of {ev_name} ***')
    return vehicle_df


def cluster_scenario(scenario_dir: Path, input_type: str, **kwargs):
    """cluster.

    :param scenario_dir Path:
    :param input_data str: Can be either 'spatial_clustered_traces'
        or 'spatial_filtered_traces'. If 'spatial_clustered_traces' is chosen,
        the temporal clustering algorithm will run on *all* clusters generated
        from the spatial clustering algorithm. If 'filtered_clustered_traces'
        is chosen, then only the clusters that fall in the geographic boundary
        will be processed.

    credits
    -------
    https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    """

    auto_run = kwargs.get('auto_run', False)

    if not auto_run:
        _ = input("Would you like to (re-)generate the time-clusters? [y]/n")
        regenerating_clusters = False if _.lower() == 'n' else True
    else:
        regenerating_clusters = True

    # FIXME: The logging module is not saving the log...
    # Setup File handler
    file_handler = logging.FileHandler(
            scenario_dir.joinpath('Temporal_Clusters', 'errors.log'), mode='w')
    file_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger('root')
    root_logger.addHandler(file_handler)

    # # create console handler and set level to INFO
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)

    # app_log = logging.getLogger('root')
    # app_log.addHandler(ch)

    time_clustered_df: pd.DataFrame = None
    if (regenerating_clusters):
        trace_dfs_and_names = [*_gen_trace_dfs(scenario_dir, input_type)]

        ### IF NOT DEBUGGING: ###  XXX I think this needs to be deleted.
                                #  cyclometric complexity is too high for
                                #  multi-threading apparently...
        # args = zip([trace_df for trace_df, _ in trace_dfs_and_names],
        #            [ev_name for _, ev_name in trace_dfs_and_names],
        #            repeat(scenario_dir, len(trace_dfs_and_names)))
        # with Pool() as p:
        #     vehicle_dfs = list(p.starmap(_cluster_ev, args))

        ### IF DEBUGGING: ###
        vehicle_dfs = []
        for trace_df, ev_name in trace_dfs_and_names:
            vehicle_dfs.append(_cluster_ev(trace_df, ev_name, scenario_dir,
                                           **kwargs))

        time_clustered_df = pd.concat(vehicle_dfs)
        time_clustered_df_groupby = time_clustered_df.groupby(
            ['Vehicle_Name', 'Space_Cluster', 'Time_Cluster'],
            group_keys=False)
        time_clustered_df = time_clustered_df_groupby.apply(
            lambda group: group.sort_values(by='Stop_Duration',
                                            ascending=False)
        )
        time_clustered_df_top_5 = time_clustered_df_groupby.apply(
            lambda group: group.sort_values(by='Stop_Duration',
                                            ascending=False).head(5)
        )
        save_file = scenario_dir.joinpath('Temporal_Clusters',
                                          'Clustered_Data',
                                          'time_clustered_full.csv')
        time_clustered_df.to_csv(save_file)
        save_file = scenario_dir.joinpath('Temporal_Clusters',
                                          'Clustered_Data',
                                          'time_clustered_top5.csv')
        time_clustered_df_top_5.to_csv(save_file)

    else:
        load_file = scenario_dir.joinpath('Temporal_Clusters',
                                          'Clustered_Data',
                                          'time_clustered_full.csv')
        time_clustered_df = pd.read_csv(load_file)
        # Reset index
        time_clustered_df = time_clustered_df.set_index(
            ['Vehicle_Name', 'Space_Cluster', 'Time_Cluster', 'Row_No'])

    if type(time_clustered_df) != pd.DataFrame:
        raise TypeError("time_clustered_df is not of type 'pd.DataFrame'")

    def _summary_table(time_clustered_df: pd.DataFrame):
        # Get the centroid duration and arrival time for each time-cluster
        time_clustered_df_groupby = time_clustered_df.groupby(
            ['Vehicle_Name', 'Space_Cluster', 'Time_Cluster'])
        time_clustered_df = time_clustered_df_groupby.mean()

        # Rename columns
        time_clustered_df.rename({'Stop_Arrival': 'mean_Stop_Arrival',
                                  'Stop_Duration': 'mean_Stop_Duration'})

        # Add column for standard deviation around each mean
        time_clustered_df[['std_Stop_Arrival', 'std_Stop_Duration']] = (
            time_clustered_df_groupby.std()[['Stop_Arrival', 'Stop_Duration']]
        )

        # Add column for number of datapoints constituting each mean
        time_clustered_df['Count'] = time_clustered_df_groupby.count()[
            'Stop_Duration']

        # Filter out means with less than `min_count` values.
        min_count = 2
        # min_count = 20  # XXX Uncomment me.
        time_clustered_df = time_clustered_df[
            time_clustered_df['Count'] >= min_count]

        # Filter out datapoints with duration < `min_hours`
        min_hours = 0.1
        #min_hours = 0.5  # XXX Uncomment me.
        time_clustered_df = time_clustered_df[
            time_clustered_df['Stop_Duration'] >= min_hours]

        # Sort spatial clusters by duration of longest temporal-cluster:
        time_clustered_df = time_clustered_df.\
            reset_index().\
            sort_values(['Vehicle_Name', 'Stop_Duration'],
                        ascending=[True, False])

        # Filter out datapoints which are in outlier time-clusters.
        time_clustered_df = time_clustered_df[
            time_clustered_df['Time_Cluster']!=-1
        ].set_index(['Vehicle_Name', 'Space_Cluster', 'Time_Cluster'])


        # Get the top 5 time-cluster means of each space-cluster
        # TODO It has an unexpected output
        # time_clustered_df = time_clustered_df.groupby(
        #     ['Vehicle_Name', 'Space_Cluster', 'Time_Cluster']).head(5)

        # Index
        # idx = tdf_sum.groupby(
        #     ['Vehicle_Name','Space_Cluster'])[
        #         'Stop_Duration'].max().sort_values(ascending=False).index
        # tdf_sum.reindex(index=idx, level=['Vehicle_Name', 'Space_Cluster'])

        return time_clustered_df

    save_dir = scenario_dir.joinpath('Temporal_Clusters', 'Clustered_Data')
    save_file = save_dir.joinpath('time_clustered_summary.csv')
    time_clustered_df_summary = _summary_table(time_clustered_df)
    time_clustered_df_summary.to_csv(save_file)
    save_file = save_dir.joinpath('time_clustered_summary.txt')
    time_clustered_df_summary.to_string(save_file)
    save_file = save_dir.joinpath('time_clustered_summary.html')
    time_clustered_df_summary.to_html(save_file)
