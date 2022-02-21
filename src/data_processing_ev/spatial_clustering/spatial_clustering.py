"""
Create clusters to simplify GPS traces.

In the context of taxis, these clusters are equivalent to taxi-stops. By
modifying `eps` and `radius`, the DBSCAN clustering algorithm's sensitivity
can be tuned.

Inputs:
    - Raw gps traces (CSV).
    - Model parameters
Outputs:
    - ./Clusters:
        - Folium map of clustered datapoints. Each cluster is a unique colour.
          (HTML)
        - Folium map of cluster means. (HTML)
        - Route file with preliminary cluster-to-cluster trips. (XML)
            - This is legacy. To remove.  # TODO
    - ./Filter_Data/Trace_Clustered:
        - GPS traces with ordered clusters, rather than *each and every* data-
          point. (CSV)

Author: Chris Abraham
"""


from collections import Counter
import random
from pathlib import Path
import pandas as pd
from sklearn.cluster import OPTICS  # OPTICS clustering algorithm.
from sklearn.preprocessing import StandardScaler
import folium
from typing import Optional
from multiprocessing import Pool, cpu_count
from itertools import repeat
# Deprecated imports:
# import hdbscan  # HDBSCAN clustering algorithm. (Hopefully better.)
# from folium.plugins import Search
# from tqdm import tqdm


def map_scatter(df, map, colors=[0]):
    """
    Generate a heatmap of coordinates from `df` on `map`.

    Parameters:
        df (dataframe): dataframe with coordinates ordered over time
        map (map): folium map object
        colors (int array):  array indicating which cluster each data point in
            df belongs to.
    """
    # generating random colors for each of the cluster codes in `colors`
    if len(colors):  # Do following code only if `colors` not empty
        num_colors = max(colors) + 1
        color_arr = []
        if num_colors > 1:
            for _ in range(num_colors):
                rand_int = random.randint(0, 16777215)
                rand_hex = str(hex(rand_int))
                rand_color = '#' + rand_hex[2:]
                color_arr.append(rand_color)
        # else generating all colors as black
        else:
            color_arr.append("black")
            for _ in range(len(df)):
                colors.append(0)

        # for each row in the traces dataset, plot the corresponding latitude
        # and longitude on the map
        colors_enum = enumerate(colors)
        for i, row in df.iterrows():
            cluster_num = colors_enum.__next__()[1]
            color = color_arr[cluster_num]
            if color != "black":
                folium.CircleMarker((row.Latitude, row.Longitude), radius=5,
                                    weight=1, color='black', fill_color=color,
                                    fill_opacity=1,
                                    popup=folium.Popup(f'*{cluster_num}*',
                                                       show=True)
                                    ).add_to(map)
            else:
                folium.CircleMarker((row.Latitude, row.Longitude), radius=1,
                                    weight=1, color='black', fill_color=color,
                                    fill_opacity=1).add_to(map)


class Dbscan_Algorithm:
    """Run functions in the order defined in class definition."""

    def __init__(self, scenario_dir: Path, df: pd.DataFrame,
                 regenerate_model: bool, saved_model: Path):
        self.original_df = df
        #   Prepare data for model
        lat_lon = df[['Latitude', 'Longitude']]
        lat_lon = lat_lon.values.astype('float32', copy=False)
        #   Normalise data
        lat_lon_data_scaler = StandardScaler().fit(lat_lon)
        lat_lon = lat_lon_data_scaler.transform(lat_lon)
        #   Construct model
        if not regenerate_model:
            self.model_labels = pd.read_csv(saved_model)['Cluster'].to_numpy()
        else:
            # Note: Approx 100km per degree Longitude/Latitude.
            # DBSCAN Version: Speed, not RAM optimised.
            """
                # 0.0002 deg ~= 0.02 km = 20 m
            self.model = DBSCAN(eps=0.0002, min_samples=70,
                                metric='euclidean', n_jobs=-1).fit(lat_lon)
            """
            # OPTICS Version: RAM, not Speed optimised.
            #   0.0002 deg ~= 0.02 km = 20 m
            #   Increased min_samples from 70 to 700 since we are using
            #   10 taxis' data. TODO: Expose these parameters as cons-
            #   tructor arguments.
            # XXX: I am changing min_samples from 70 to 10. Kampala has less
            # data than the Stell_SSW_BF scenario.
            self.model = OPTICS(min_samples=10, max_eps=0.0002,
                                metric='euclidean', cluster_method='dbscan',
                                n_jobs=1).fit(lat_lon)
            # self.model = hdbscan.HDBSCAN(
            #     min_cluster_size=500,
            #     min_samples=50  # , cluster_selection_epsilon=0.001
            # ).fit(lat_lon)
            self.model_labels = self.model.labels_
            #   Save the model
            saved_model.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.model_labels, columns=['Cluster']).to_csv(
                saved_model, index=False)
        #   Seperate outliers from clustered data
        self.outliers_df = df[self.model_labels == -1]
        self.clusters_df = df[self.model_labels != -1]
        df['Cluster'] = self.model_labels
        self.full_df = df

    def print_summary(self):
        # Get info about the clusters
        clusters_stats = Counter(self.model_labels)
        print(clusters_stats)
        print(self.outliers_df.head())
        print("Number of clusters = {}".format(len(clusters_stats) - 1))

    # TODO Make this function do the functionality of `map_cluster_means` AND
    #   `plot_original_coords`
    def map_results(self, df: Optional[pd.DataFrame] = None):
        if not df:
            # plot original if df not specified.
            df = self.original_df
        # colors of datapoints
        colors = self.model_labels
        colors_clusters = colors[colors != -1]
        #   Plot clusters and outliers
        #   FIXME: Don't hard-code map coords
        self.stell_map_clusters = folium.Map(location=[-33.9331, 18.8668],
                                             titles='Stellenbosch',
                                             zoom_start=14)
        # Scatter every `n`th data-point onto the folium map.
        n = 1000
        map_scatter(self.clusters_df[::n], self.stell_map_clusters,
                    colors=colors_clusters)

    def generate_cluster_means(self):
        # Get mean of each cluster
        num_clusters = max(self.model_labels) + 1
        cluster_means = []
        for i in range(num_clusters):
            cluster_df = self.original_df[self.model_labels == i]
            cluster_means.append(
                (cluster_df.loc[:, "Latitude"].mean(),
                 cluster_df.loc[:, "Longitude"].mean()))
        self.cluster_means_df = pd.DataFrame(cluster_means,
                                             columns=["Latitude",
                                                      "Longitude"])

    def map_cluster_means(self):
        # colors of datapoints
        colors = self.model_labels
        colors_clusters = set(colors[colors != -1])
        # Plot cluster means
        self.stell_map_means = folium.Map(location=[-33.9331, 18.8668],
                                          titles='Stellenbosch', zoom_start=14)
        map_scatter(self.cluster_means_df, self.stell_map_means,
                    colors=colors_clusters)


def cluster_traces(trace_file: Path, scenario_dir: Path,
                   regenerate_model: bool, saved_model_dir: Path) -> None:
    # Convert gps trace to pandas dataframe
    trace_df = pd.read_csv(str(trace_file))
    ev_name = trace_file.stem

    # DBSCAN Algorithm
    print(f"### Starting clustering of {ev_name} ###")

    saved_model = saved_model_dir.joinpath(
        f"model_labels_{ev_name}.csv")
    dbscan_algorithm = Dbscan_Algorithm(scenario_dir, trace_df,
                                        regenerate_model, saved_model)

    #   Visualise Results
    #       print summary
    # dbscan_algorithm.print_summary()
    #       plot clusters
    dbscan_algorithm.map_results()
    #   Get mean of each cluster
    dbscan_algorithm.generate_cluster_means()
    #   Plot cluster means
    dbscan_algorithm.map_cluster_means()

    # Save maps
    plot_file = scenario_dir.joinpath('Spatial_Clusters', 'Maps',
                                      'Clustered_Means',
                                      f'{ev_name}.html')
    dbscan_algorithm.stell_map_means.save(str(plot_file))
    plot_file = scenario_dir.joinpath('Spatial_Clusters', 'Maps',
                                      'Clustered_Datapoints',
                                      f'{ev_name}.html')
    dbscan_algorithm.stell_map_clusters.save(str(plot_file))

    # TODO make this path an input argument
    # export ordered clusters as csv
    export_file = scenario_dir.joinpath('Spatial_Clusters', 'Clustered_Traces',
                                        f'{ev_name}.csv')
    dbscan_algorithm.full_df.to_csv(export_file, index=False)

    print(f"### Completed clustering of {ev_name} ###")


def copy_datapoints_unclustered(trace_file: Path, scenario_dir: Path):
    """ Copy the input gps traces into the clustered_traces directory, without
    clustering. All points will be part of "Cluster 1". """

    with open(trace_file, 'r') as f_in:
        with open(scenario_dir.joinpath('Spatial_Clusters', 'Clustered_Traces',
                                        f'{trace_file.stem}.csv'
                                        ), 'w+') as f_out:
            header = f_in.readline()[:-1]  # Read the header line, discarding
                                            # the line-break character.
            header += ",Cluster\n"
            f_out.write(header)
            for line in f_in.readlines():
                line = line[:-1]
                line += ",0\n"
                f_out.write(line)
    return


def cluster_scenario(scenario_dir: Path, skip: bool = False, **kwargs):
    """
    Create ordered clusters from the gps traces and export as csv.

    Inputs:
        scenario_dir: Path object pointing to the scenario's root directory.
        skip: If this is True, the clusering process will be skipped, and all
              datapoints will be labelled as "cluster 1".
    """
    # # If using reduced traces:
    # # TODO: Decide if you're doing this or not... Did you do this in Stell
    # # scenario? I didn't, but I planned to. _Maybe_ it will give better
    # # results.
    # # FIXME: Generate Reduced_Traces automatically.
    # trace_files = [*scenario_dir.joinpath('Spatial_Clusters',
    #                                       'Reduced_Traces').glob('*.csv')]
    # # FIXME: I've temporarily implemented this in the input-traces data pre-
    # # processing step. That's probably not the best way to do it though...

    auto_run = kwargs.get('auto_run', False)

    trace_files = sorted(
        [*scenario_dir.joinpath('_Inputs', 'Traces',
                                'Processed').glob('*.csv')]
    )
    if skip:
        with Pool(cpu_count() - 1) as p:
            args = zip(trace_files,
                       repeat(scenario_dir, len(trace_files)))
            p.starmap(copy_datapoints_unclustered, args)
    if not skip:
        saved_model_dir = scenario_dir.joinpath('Spatial_Clusters',
                                                'Clustered_Traces', '.cache')
        # Check if saved_model exists. If it does, the user will have the
        # option to either use this file or to regenerate it.
        regenerate_model = True
        saved_models_exist = any(saved_model_dir.iterdir())
        if saved_models_exist:
            print("Model labels already found at: \n\t" +
                  str(saved_model_dir.absolute()))
            if not auto_run:
                _ = input("Would you like to use these files (otherwise " +
                          "program will regenerate models)? [y]/n")
                regenerate_model = True if _.lower() == 'n' else False
            else:
                print("Using existing model labels...")

        # TODO: Deprecate the below code. Combined traces need to be manually
            # constructed (I think -- can't remember now).
        # trace_dfs = [pd.read_csv(trace_file) for trace_file in trace_files]
        # traces_combined = pd.concat(trace_dfs, keys=[trace_file.stem for
        #                                              trace_file in trace_files])
        # cluster_traces(traces_combined, scenario_dir, regenerate_model,
        #                saved_model_dir)

        # TODO Make debugging a function parameter.

        # IF DEBUGGING:
        # -------------
        # for trace_file in tqdm(trace_files):
        #     cluster_traces(trace_file, scenario_dir, regenerate_model,
        #                    saved_model_dir)
        # IF NOT DEBUGGING:
        # ----------------- XXX
        with Pool(cpu_count() - 1) as p:
            args = zip(trace_files,
                       repeat(scenario_dir, len(trace_files)),
                       repeat(regenerate_model, len(trace_files)),
                       repeat(saved_model_dir, len(trace_files)))
            p.starmap(cluster_traces, args)
