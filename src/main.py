#!/usr/bin/env python3
"""This program runs all the components of the data_processing_ev package."""

import data_processing_ev as dpr
from pathlib import Path
from typing import Iterable, SupportsFloat
import argparse


def get_input_data_fmt(scenario_dir: Path):
    """
    Identifies the input data format by analysing the
    `scenario_dir`/_Inputs/Traces directory.

    Currently, the formats in `dpr.DATA_FMTS` are supported.
    """

    if len([*scenario_dir.joinpath(
            '_Inputs', 'Traces', 'Original').glob('GTFS*')]):
        print("GTFS inputs detected...")
        data_fmt = dpr.DATA_FMTS['GTFS']
    else:
        print("Assuming GPS inputs...")
        data_fmt = dpr.DATA_FMTS['GPS']

    return data_fmt


def run(scenario_dir: Path, steps: Iterable[SupportsFloat],
        configuring_steps: bool = False, **kwargs):
    """Run specified steps of data_analysis."""
    auto_run = kwargs.get('auto_run', False)

    kwargs['input_data_fmt'] = get_input_data_fmt(scenario_dir)

    if 0 in steps:
        dpr.initialise_scenario(scenario_dir, **kwargs)
    if 1 in steps or 1.1 in steps:
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

            dpr.data_visualisation.map_scenario(
                scenario_dir, mapping_points, mapping_lines, mapping_heatmap,
                mapping_geojson, saving=True, **kwargs)
        else:
            dpr.data_visualisation.map_scenario(scenario_dir, **kwargs)
    if 1 in steps or 1.2 in steps:
        dpr.data_visualisation.animate_scenario(scenario_dir, **kwargs)
    if 2 in steps or 2.1 in steps:
        """Spatial clustering"""
        if not auto_run:
            _ = input("Would you like to label *all* datapoints as part of " +
                      "*one* big cluster? [y]/n ")
            skip = False if _.lower() == 'n' else True
        else:
            skip = True
        dpr.spatial_clustering.cluster_scenario(scenario_dir, skip=skip,
                                                **kwargs)
    if 2 in steps or 2.2 in steps:
        """Spatial filtering"""
        dpr.spatial_filtering.filter_scenario(scenario_dir, **kwargs)
    if 2 in steps or 2.3 in steps:
        """List the dates which survived spatial filtering."""
        dpr.save_dates_remaining.save_dates_remaining(scenario_dir, **kwargs)
    if 3 in steps or 3.1 in steps:
        """Stop extraction"""
        dpr.stop_extraction.extract_stops(scenario_dir, **kwargs)
    if 3 in steps or 3.2 in steps:
        """Stop duration box plots"""
        # TODO: GTFS implementation should consider frequencies.txt.
        dpr.stop_duration_box_plots.plot_stop_duration_boxes(
            scenario_dir, plot_blotches=False, **kwargs)
    if 3 in steps or 3.3 in steps:
        """Temporal clustering"""
        #   TODO: Disable this step by default (similar to how we disable
        # spatial_clustering).
        if kwargs['input_data_fmt'] == dpr.DATA_FMTS['GPS']:
            clustering_type = 'spatial_filtered_traces'
            dpr.temporal_clustering.cluster_scenario(scenario_dir, clustering_type,
                                                     **kwargs)
        elif kwargs['input_data_fmt'] == dpr.DATA_FMTS['GTFS']:
            print("Warning: Temporal clustering is not implemented for GTFS " +
                  "scenarios yet.")
        else:
            raise ValueError(dpr.DATA_FMT_ERROR_MSG)

    if 4 in steps:
        """Routing"""
        dpr.routing.build_routes(scenario_dir, **kwargs)
    if 5 in steps or 5.1 in steps:
        """Simulation"""
        dpr.ev_simulation.simulate_all_routes(
            scenario_dir, skip_existing=False, **kwargs)
    if 5 in steps or 5.2 in steps:
        # De-combine simulation results.
        dpr.results_splitter.split_results(scenario_dir, **kwargs)
    if 6 in steps or 6.1 in steps:
        """Generate Plots and Statistics from EV Simulation Results"""
        dpr.ev_results_analysis.run_ev_results_analysis(scenario_dir, **kwargs)
        # TODO: GTFS implementation of ev_box_plots should consider
        # frequencies.txt.
        dpr.ev_box_plots.plot_ev_energy_boxes(
            scenario_dir, **kwargs)
    if 6 in steps or 6.2 in steps:
        """Generate Plots and Statistics from PV Simulation Results"""
        # dpr.pv_results_analysis.run_pv_results_analysis(scenario_dir, **kwargs)
        if kwargs['input_data_fmt'] == dpr.DATA_FMTS['GPS']:
            dpr.pv_results_analysis.run_pv_results_analysis(scenario_dir, **kwargs)
        elif kwargs['input_data_fmt'] == dpr.DATA_FMTS['GTFS']:
            print("Warning: PV Results Analysis is not implemented for GTFS " +
                  "scenarios yet.")
        else:
            raise ValueError(dpr.DATA_FMT_ERROR_MSG)


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

    while scenario_dir.exists() is False:
        print("The path does not exist!")
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

    _ = input("Would you like to skip all prompts and use only default " +
              "values? y/[n] ")
    auto_run = True if _.lower() == 'y' else False

    if auto_run:
        conf = False
    else:
        _ = input("Would you like to configure the steps of the " +
                  "analysis? y/[n] ")
        conf = True if _.lower() == 'y' else False

    kwargs = {'auto_run': auto_run, 'input_data_fmt': None}

    # Run all the steps
    # run(scenario_dir, steps=range(6), configuring_steps=conf)
    run(scenario_dir, steps=steps, configuring_steps=conf, **kwargs)
