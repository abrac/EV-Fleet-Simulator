# TODO Maybe convert the dir structure to format in
# https://stackoverflow.com/a/62570479/10462623 OR Create the dir structure in
# the file system, and store it using Path.glob('**')

DATA_FMTS = {'GPS': 1, 'GTFS': 2}

EV_MODELS = {'SUMO': 1, 'Hull': 2}

from .scenario_initialisation.scenario_initialisation \
    import initialise_scenario
from . import data_visualisation  # TODO Make this consistent with the others.
from .spatial_clustering import spatial_clustering
from .spatial_clustering import spatial_filtering
from .spatial_clustering import save_dates_remaining
from .temporal_clustering import temporal_clustering
from .temporal_clustering import stop_extraction
from .temporal_clustering import stop_duration_box_plots
from .routing import routing
from .ev_simulation import sumo_ev_simulation
from .ev_simulation import hull_ev_simulation
from .ev_simulation import results_splitter
from .results_analysis import ev_results_analysis
from .results_analysis import ev_box_plots
from .results_analysis import pv_results_analysis
from .results_analysis import wind_results_analysis

from pathlib import Path
from typing import Iterable, SupportsFloat
import argparse
import time
import subprocess
import multiprocessing as mp


DATA_FMT_ERROR_MSG = "The function has not been implemented for the " + \
                     "currently selected input data format."

PIGZ_WARNING_ACKNOWLEDGED = False

SCENARIO_DIR_STRUCTURE = {
    '_Inputs': {
        'Traces': {
            'Original': None,
            'Processed': None
        },
        'Map': {
            'Boundary': None,
            'Construction': None
        },
        'Configs': None,
        'Weather': None
    },
    'Data_Visualisations': {
        'Maps': None,
        'Route_Animations': None
    },
    'Spatial_Clusters': {
        'Reduced_Traces': None,
        'Clustered_Traces': {
            '.cache': None
        },
        'Filtered_Traces': None,
        'Maps': {
            'Clustered_Datapoints': None,
            'Clustered_Means': None
        },
    },
    'Temporal_Clusters': {
        'Clustered_Data': None,
        'Graphs': None,
        'Stop_Labels': None
    },
    'Routes': {
        'Routes': None,
        'Trips': None
    },
    'EV_Simulation': {
        'Sumocfgs': None,
        'SUMO_Simulation_Outputs': None,
        'Hull_Simulation_Outputs': None,
            # Simulation outputs from our custom simulation model, developed by
            # Christopher Hull. [TODO Cite.]
            # TODO: Only allow one to be selected!
    },
    'SAM_Simulation': {
        'Results': None,
        'SAM_Scenario_File': None
    },
    'EV_Results': {
    }
}

MODULES = """
0. scenario_initialisation
1. data_visualisation
    1. mapping
    2. route_animation
    3. map_size_calculation
2. spatial_analysis
    1. spatial_clustering
    2. date_filtering_and_separation
    3. save_dates_remaining
3. temporal_analysis
    1. stop_extraction
    2. stop_duration_box_plots
    3. temporal_clustering
4. routing
5. ev_simulation
    5.1. sumo_ev_simulation
    5.2. result_splitting
    5.3. hull_ev_simulation
6. results_analysis
    6.1. ev_results_analysis
    6.2. pv_results_analysis
    6.3. wind_results_analysis
"""


def decompress_file(file: Path) -> Path:
    try:
        gz_index = file.name.find('.gz')
        # if the file ends with gz
        if gz_index != -1:
            decompressed_file = file.parent.joinpath(file.name[:gz_index])
            compressed_file = file
        else:
            decompressed_file = file
            compressed_file = file.parent.joinpath(file.name + '.gz')

        # If the file has already been decompressed...
        if decompressed_file.exists():
            if compressed_file.exists():
                _ = input("Both the compressed and the decompressed " +
                          "versions of the file exist. May I delete one? [y]/n")
                delete = True if _.lower() != 'n' else False
                if delete:
                    compressed_file.unlink()
            return decompressed_file
        else:
            p = subprocess.Popen(['pigz', '-p', str(mp.cpu_count() - 2), '-d',  # '--quiet',
                                       str(compressed_file.absolute())])
            # Wait until the decompression is complete.
            process_complete = False
            while not process_complete:
                poll = p.poll()
                if poll is None:
                    process_complete = False
                else:
                    process_complete = True
            return decompressed_file

    except subprocess.CalledProcessError:
        print("Warning: Pigz failed to compress the xml file.")
        return None

    except OSError:
        print("Warning: You probably haven't installed `pigz`. Install " +
              "it if you want the script to automagically compress your " +
              "combined XML files after it has been split!")
        global PIGZ_WARNING_ACKNOWLEDGED
        if not PIGZ_WARNING_ACKNOWLEDGED:
            input("For now, press enter to ignore.")
            PIGZ_WARNING_ACKNOWLEDGED = True
        return None


def compress_file(file: Path) -> Path:
    try:
        gz_index = file.name.find('.gz')
        # if the file ends with gz
        if gz_index != -1:
            decompressed_file = file.parent.joinpath(file.name[:gz_index])
            compressed_file = file
        else:
            decompressed_file = file
            compressed_file = file.parent.joinpath(file.name + '.gz')

        if compressed_file.exists():
            if decompressed_file.exists():
                _ = input("Both the compressed and the decompressed " +
                          "versions of the file exist. May I delete one? [y]/n")
                delete = True if _.lower() != 'n' else False
                if delete:
                    decompressed_file.unlink()
            return compressed_file
        else:
            p = subprocess.Popen(['pigz', '-p', str(mp.cpu_count() - 2),  # '--quiet',
                                 str(decompressed_file.absolute())])

            # Wait until the compression is complete.
            process_complete = False
            while not process_complete:
                poll = p.poll()
                if poll is None:
                    process_complete = False
                else:
                    process_complete = True

            return compressed_file

    except subprocess.CalledProcessError:
        print("Warning: Pigz failed to compress the xml file.")
        return None
    except OSError:
        print("Warning: You probably haven't installed `pigz`. Install " +
              "it if you want the script to automagically compress your " +
              "combined XML files after it has been split!")
        global PIGZ_WARNING_ACKNOWLEDGED
        if not PIGZ_WARNING_ACKNOWLEDGED:
            input("For now, press enter to ignore.")
            PIGZ_WARNING_ACKNOWLEDGED = True
        return None


def get_input_data_fmt(scenario_dir: Path):
    """
    Identifies the input data format by analysing the
    `scenario_dir`/_Inputs/Traces directory.

    Currently, the formats in `DATA_FMTS` are supported.
    """

    if len([*scenario_dir.joinpath(
            '_Inputs', 'Traces', 'Original').glob('GTFS*')]):
        print("GTFS inputs detected...")
        data_fmt = DATA_FMTS['GTFS']
    else:
        print("Assuming GPS inputs...")
        data_fmt = DATA_FMTS['GPS']

    return data_fmt


def _run(scenario_dir: Path, steps: Iterable[SupportsFloat],
        configuring_steps: bool = False, **kwargs):
    """Run specified steps of data_analysis."""
    auto_run = kwargs.get('auto_run', False)

    kwargs['input_data_fmt'] = get_input_data_fmt(scenario_dir)
    # TODO Other important kwargs: EV_Model

    if 0 in steps:
        initialise_scenario(scenario_dir, **kwargs)
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

            data_visualisation.map_scenario(
                scenario_dir, mapping_points, mapping_lines, mapping_heatmap,
                mapping_geojson, saving=True, **kwargs)
        else:
            data_visualisation.map_scenario(scenario_dir, **kwargs)
    if 1 in steps or 1.2 in steps:
        data_visualisation.animate_scenario(scenario_dir, **kwargs)
    if 1 in steps or 1.3 in steps:
        data_visualisation.get_map_size(scenario_dir, **kwargs)
    if 2 in steps or 2.1 in steps:
        """Spatial clustering"""
        if not auto_run:
            _ = input("Would you like to label *all* datapoints as part of " +
                      "*one* big cluster? [y]/n ")
            skip = False if _.lower() == 'n' else True
        else:
            skip = True
        spatial_clustering.cluster_scenario(scenario_dir, skip=skip,
                                                **kwargs)
    if 2 in steps or 2.2 in steps:
        """Spatial filtering"""
        spatial_filtering.filter_scenario(scenario_dir, **kwargs)
    if 2 in steps or 2.3 in steps:
        """List the dates which survived spatial filtering."""
        save_dates_remaining.save_dates_remaining(scenario_dir, **kwargs)
    if 3 in steps or 3.1 in steps:
        """Stop extraction"""
        stop_extraction.extract_stops(scenario_dir, **kwargs)
    if 3 in steps or 3.2 in steps:
        """Stop duration box plots"""
        # TODO: GTFS implementation should consider frequencies.txt.
        stop_duration_box_plots.plot_stop_duration_boxes(
            scenario_dir, plot_blotches=False, **kwargs)
    if 3 in steps or 3.3 in steps:
        """Temporal clustering"""
        #   TODO: Disable this step by default (similar to how we disable
        # spatial_clustering).
        print("Temporal clustering has been temporarily disabled.")
        temporal_clustering_enabled = False
        if temporal_clustering_enabled:
            if kwargs['input_data_fmt'] == DATA_FMTS['GPS']:
                clustering_type = 'spatial_filtered_traces'
                temporal_clustering.cluster_scenario(scenario_dir, clustering_type,
                                                         **kwargs)
            elif kwargs['input_data_fmt'] == DATA_FMTS['GTFS']:
                print("Warning: Temporal clustering is not implemented for GTFS " +
                      "scenarios yet.")
            else:
                raise ValueError(DATA_FMT_ERROR_MSG)

    if 4 in steps:
        """Routing"""
        routing.build_routes(scenario_dir, **kwargs)
    if 5 in steps or 5.1 in steps:
        """Simulation"""
        sumo_ev_simulation.simulate_all_routes(
            scenario_dir, skip_existing=False, **kwargs)
    if 5 in steps or 5.2 in steps:
        # De-combine simulation results.
        results_splitter.split_results(scenario_dir, **kwargs)
    if 5 in steps or 5.3 in steps:
        hull_ev_simulation.simulate(scenario_dir, **kwargs)
    if 6 in steps or 6.1 in steps:
        """Generate Plots and Statistics from EV Simulation Results"""
        ev_results_analysis.run_ev_results_analysis(scenario_dir, **kwargs)
        # TODO: GTFS implementation of ev_box_plots should consider
        # frequencies.txt.
        ev_box_plots.plot_ev_energy_boxes(
            scenario_dir, **kwargs)
    if 6 in steps or 6.2 in steps:
        """Generate Plots and Statistics from PV Simulation Results"""
        if kwargs['input_data_fmt'] == DATA_FMTS['GPS']:
            pv_results_analysis.run_pv_results_analysis(scenario_dir, **kwargs)
        elif kwargs['input_data_fmt'] == DATA_FMTS['GTFS']:
            print("Warning: PV Results Analysis is not implemented for GTFS " +
                  "scenarios yet.")
        else:
            raise ValueError(DATA_FMT_ERROR_MSG)
    if 6 in steps or 6.3 in steps:
        """Generate Plots and Statistics from Wind Simulation Results"""
        if kwargs['input_data_fmt'] == DATA_FMTS['GPS']:
            wind_results_analysis.run_wind_results_analysis(scenario_dir, **kwargs)
        elif kwargs['input_data_fmt'] == DATA_FMTS['GTFS']:
            print("Warning: Wind Results Analysis is not implemented for GTFS" +
                  "scenarios yet.")
        else:
            raise ValueError(DATA_FMT_ERROR_MSG)


def main():
    # TODO get scenario_dir via terminal arguments

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'scenario_dir', nargs='?', default=None, metavar='str',
        help='The root directory of the scenario to be simulated.')
    parser.add_argument(
        '--steps', default=None, metavar='List[float]',
        help="The steps to be run on the scenario as a comma-seperated list " +
             " of floats without spaces (e.g. '1,2.2,4').")
    parser.add_argument(
        '--debug', action='store_true')
    parser.add_argument(
        '--incl-weekends', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb
        pdb.set_trace()

    if args.scenario_dir:
        scenario_dir = Path(args.scenario_dir)
    else:
        _ = input("Specify scenario root directory: ")
        scenario_dir = Path(_)

    while scenario_dir.exists() is False:
        print("The path does not exist!")
        _ = input("Specify scenario root directory: ")
        scenario_dir = Path(_)

    while True:
        if args.steps:
            steps_str = args.steps
        else:
            print("Available steps: ")
            print(MODULES)
            steps_str = input("Specify steps to be run as a comma-seperated " +
                              "list of floats without spaces (e.g. '1,2.2,4'): ")
        try:
            steps = [float(step) for step in steps_str.split(',')]
        except ValueError:
            print("Error: That is not a valid selection! Try again...")
            time.sleep(1)
            continue
        break

    # TODO: Implement auto_run properly!!!
    # TODO And delete the conf parameter!!!

    # Hard-coding these values until they are properly implemented!

    # _ = input("Would you like to skip all prompts and use only default " +
    #           "values? y/[n] ")
    # auto_run = True if _.lower() == 'y' else False

    # if auto_run:
    #     conf = False
    # else:
    #     _ = input("Would you like to configure the steps of the " +
    #               "analysis? y/[n] ")
    #     conf = True if _.lower() == 'y' else False

    auto_run = False
    conf = False

    kwargs = {'auto_run': auto_run, 'input_data_fmt': None, 'incl_weekends': args.incl_weekends}

    # Run all the steps
    # run(scenario_dir, steps=range(6), configuring_steps=conf)
    _run(scenario_dir, steps=steps, configuring_steps=conf, **kwargs)
