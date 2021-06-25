#!/usr/bin/env python3
"""This program runs all the components of the data_processing_ev package."""

import data_processing_ev as dpr
from pathlib import Path
from typing import Iterable, SupportsFloat
import argparse


def run(scenario_dir: Path, steps: Iterable[SupportsFloat],
        configuring_steps: bool = False):
    """Run specified steps of data_analysis."""
    if 0 in steps:
        dpr.initialise_scenario(scenario_dir)
    if 1 in steps:
        if configuring_steps:
            _ = input("Plot gps points? [y]/n")
            mapping_points = True if _.lower() != 'n' else False
            _ = input("Plot lines connecting the points? y/[n]")
            mapping_lines = False if _.lower() != 'y' else True
            _ = input("Plot heatmap clustering the points? [y]/n")
            mapping_heatmap = True if _.lower() != 'n' else False
            # Get map features (taz boundaries, etc.)
            geojson_filelist = scenario_dir.joinpath('0_Inputs',
                                                     'Map').glob('*.geojson')
            if geojson_filelist != []:
                _ = input("Add map features from geojson? [y]/n")
                mapping_geojson = True if _.lower() != 'n' else False
            else:
                mapping_geojson = False
            _ = input("Animate gps points? [y]/n")
            animating_points = True if _.lower() != 'n' else False

            dpr.data_visualisation.map_scenario(
                scenario_dir, mapping_points, mapping_lines, mapping_heatmap,
                mapping_geojson, saving=True)
            if animating_points:
                dpr.data_visualisation.animate_scenario(scenario_dir)
        else:
            dpr.data_visualisation.map_scenario(scenario_dir)
    if 2 in steps or 2.1 in steps:
        """Spatial clustering"""
        _ = input("Would you like to label *all* datapoints as part of " +
                  "*one* big cluster? y/[n] ")
        skip = True if _.lower() == 'y' else False
        dpr.spatial_clustering.cluster_scenario(scenario_dir, skip=skip)
    if 2 in steps or 2.2 in steps:
        """Spatial filtering"""
        dpr.spatial_filtering.filter_scenario(scenario_dir)
    if 3 in steps:
        """Temporal clustering"""
        if configuring_steps:
            _ = input("For temporal clustering, use geographically filtered " +
                      "traces? [y]/n\n\t")
            clustering_type = ('spatial_clustered_traces' if _.lower() != 'n'
                               else 'spatial_filtered_traces')
        else:
            clustering_type = 'spatial_filtered_traces'
        dpr.temporal_clustering.cluster_scenario(scenario_dir, clustering_type)
    if 4 in steps:
        """Routing"""
        dpr.routing.build_routes(scenario_dir)
    if 5 in steps:
        """Simulation"""
        # TODO Remove below if irrelevant
        # _ = input("Would you like to skip the running of sumocfg's that " +
        #           "have already been run before? [y]/n \n\t")
        dpr.ev_simulation.simulate_all_routes(scenario_dir,
                                              skip_existing=False)
    if 6 in steps:
        """Generate Plots and Statistics from Simulation Results"""
        dpr.results_analysis.run_results_analysis(scenario_dir)


if __name__ == "__main__":
    # TODO get scenario_dir via terminal arguments

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'scenario_dir', nargs='?', default=None, metavar='str',
        help='The root directory of the scenario to be simulated.')
    parser.add_argument(
        '--steps', default=None, metavar='List[float]',
        help="The steps to be run on the scenario as a comma-seperated list " +
             " of floats without spaces (e.g. '1,2.2,4').")
    args = parser.parse_args()

    if args.scenario_dir:
        scenario_dir = Path(args.scenario_dir)
    else:
        _ = input("Specify scenario root directory: ")
        scenario_dir = Path(_)

    if args.steps:
        steps_str = args.steps
    else:
        print("Available steps: ")
        print(dpr.MODULES)
        steps_str = input("Specify steps to be run as a comma-seperated " +
                          "list of floats without spaces (e.g. '1,2.2,4'): ")
    steps = [float(step) for step in steps_str.split(',')]

    if scenario_dir.exists() is False:
        raise ValueError("The path does not exist!")
    _ = input("Would you like to configure the steps of the analysis? y/[n] ")
    conf = False if _.lower() != 'y' else True
    # Run all the steps
    #run(scenario_dir, steps=range(6), configuring_steps=conf)
    run(scenario_dir, steps=steps, configuring_steps=conf)
