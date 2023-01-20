# TODO Maybe convert the dir structure to format in
# https://stackoverflow.com/a/62570479/10462623 OR Create the dir structure in
# the file system, and store it using Path.glob('**')

DATA_FMTS = {'GPS': 0, 'GTFS': 1}

EV_MODELS = {'SUMO': 0, 'Hull': 1}

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
from .routing import fcd_conversion
from .ev_simulation import sumo_ev_simulation
from .ev_simulation import results_splitter
from .ev_simulation import hull_ev_simulation
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
import logging
from importlib.metadata import version
import sys

# loggers:
LOGGERS = {
    'main': logging.getLogger('main'),
    'input': logging.getLogger('input'),
    'sumo_output': logging.getLogger('sumo_output')  # For SUMO simulation outputs only
}

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
    '_Logs': {},
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
    'Mobility_Simulation': {
        'Routes': None,
        'Trips': None,
        'FCD_Data': None
    },
    'EV_Simulation': {
        'Sumocfgs': None,
        'EV_Simulation_Outputs': None,
    },
    'REG_Simulation': {
        'Results': None,
        'SAM_Scenario_File': None
    },
    'EV_Results': {
    }
}

# TODO Convert MODULES to a dictionary. That way, if the steps' numbers change,
# it won't affect the rest of the code!
MODULES = """
0. scenario_initialisation
1. data_visualisation
    1.1. mapping
    1.2. route_animation
    1.3. map_size_calculation
2. spatial_analysis
    2.1. spatial_clustering
    2.2. date_filtering_and_separation
    2.3. save_dates_remaining
3. temporal_analysis
    3.1. stop_extraction
    3.2. stop_duration_box_plots
    3.3. temporal_clustering
4.x. mobility_simulation
    4.1. routing  **OR** 4.2 fcd_conversion
5.x. ev_simulation
    5.1. sumo_ev_simulation **OR** 5.2 hull_ev_simulation
6. results_analysis
    6.1. ev_results_analysis
    6.2. pv_results_analysis
    6.3. wind_results_analysis
"""


def _check_routing_status(scenario_dir: Path, **kwargs):
    """If a step in 4 or 5 were selected, check which
       mobility simulation was done."""
    routing_was_done = any(scenario_dir.joinpath(
        'Mobility_Simulation', 'Routes').iterdir())
    fcd_conversion_was_done = any(scenario_dir.joinpath(
        'Mobility_Simulation', 'FCD_Data').iterdir())
    return routing_was_done, fcd_conversion_was_done

def auto_input(prompt: str, default: str, **kwargs):
    auto_run = kwargs.get('auto_run', False)
    if auto_run:
        message = prompt + default + '\n'
        print(message)
        logging.getLogger('input').info(message)
        return default
    else:
        _selection = input(prompt)
        message = prompt + _selection + '\n'
        logging.getLogger('input').info(message)
        return _selection


def decompress_file(file: Path, **kwargs) -> Path:
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
                _ = auto_input(
                    "Both the compressed and the decompressed " +
                    "versions of the file exist. May I delete one? [y]/n  ",
                    'y', **kwargs)
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
            auto_input("For now, press enter to ignore.", '', **kwargs)
            PIGZ_WARNING_ACKNOWLEDGED = True
        return None


def compress_file(file: Path, **kwargs) -> Path:
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
                _ = auto_input(
                    "Both the compressed and the decompressed " +
                    "versions of the file exist. May I delete one? [y]/n  ",
                    'y', **kwargs)
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
            auto_input("For now, press enter to ignore.", '', **kwargs)
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


def initialise_loggers(scenario_dir, **kwargs):
    # Configure logging
    for key in LOGGERS.keys():
        if kwargs.get('debug_mode'):
            fh_formatter = logging.Formatter(
                ">>> %(asctime)s - %(levelname)s - %(name)s - "
                "%(pathname)s - %(lineno)d:\n%(message)s\n")
            ch_formatter = logging.Formatter(
                '%(levelname)s: %(message)s')
        else:
            fh_formatter = logging.Formatter(
                ">>> %(asctime)s - %(levelname)s - %(name)s - "
                "%(filename)s:\n%(message)s\n")
            ch_formatter = logging.Formatter(
                '%(levelname)s: %(message)s')

        ch_log_level = logging.WARNING
        fh_log_level = logging.DEBUG

        loggingFile = scenario_dir.joinpath('_Logs', f'{key}.log')
        loggingFile.parent.mkdir(parents=True, exist_ok=True)

        LOGGERS[key].setLevel(logging.DEBUG)

        fh = logging.FileHandler(
            loggingFile, 'a' if loggingFile.exists() else 'w')
        fh.setLevel(fh_log_level)
        fh.setFormatter(fh_formatter)

        # console handler
        ch = logging.StreamHandler()
        ch.setLevel(ch_log_level)
        ch.setFormatter(ch_formatter)

        LOGGERS[key].addHandler(ch)
        LOGGERS[key].addHandler(fh)

    LOGGERS['main'].info("Starting a new log session.")
    if kwargs.get('debug_mode'):
        LOGGERS['main'].debug("Logger in debug mode.")
    LOGGERS['main'].info(f"EV-Fleet-Sim version {version('ev-fleet-sim')}")


def print_running_step(step: str):
    LOGGERS['main'].info(f"Running step {step}")
    message = f"Running step {step}"
    print("\n\n{message}")
    print('='*len(message)+'\n')


def _run(scenario_dir: Path, steps: Iterable[SupportsFloat], **kwargs):
    """Run specified steps of data_analysis."""

    kwargs['input_data_fmt'] = get_input_data_fmt(scenario_dir)

    if steps:
        initialise_loggers(scenario_dir, **kwargs)

    if 0 in steps:
        print_running_step("0: scenario_initialisation")
        initialise_scenario(scenario_dir, **kwargs)
    if 1 in steps or 1.1 in steps:
        print_running_step("1.1: mapping")
        data_visualisation.map_scenario(scenario_dir, **kwargs)
    if 1 in steps or 1.2 in steps:
        print_running_step("1.2: route_animation")
        data_visualisation.animate_scenario(scenario_dir, **kwargs)
    if 1 in steps or 1.3 in steps:
        print_running_step("1.3: map_size_calculation")
        data_visualisation.get_map_size(scenario_dir, **kwargs)
    if 2 in steps or 2.1 in steps:
        print_running_step("2.1: spatial_clustering")
        """Spatial clustering"""
        _ = auto_input(
            "Would you like to label *all* datapoints as part of " +
            "*one* big cluster? [y]/n  ", 'y', **kwargs)
        skip = False if _.lower() == 'n' else True
        spatial_clustering.cluster_scenario(scenario_dir, skip=skip,
                                                **kwargs)
    if 2 in steps or 2.2 in steps:
        print_running_step("2.2: date_filtering_and_separation")
        """Spatial filtering"""
        spatial_filtering.filter_scenario(scenario_dir, **kwargs)
    if 2 in steps or 2.3 in steps:
        print_running_step("2.3: save_dates_remaining")
        """List the dates which survived spatial filtering."""
        save_dates_remaining.save_dates_remaining(scenario_dir, **kwargs)
    if 3 in steps or 3.1 in steps:
        print_running_step("3.1: stop_extraction")
        """Stop extraction"""
        stop_extraction.extract_stops(scenario_dir, **kwargs)
    if 3 in steps or 3.2 in steps:
        print_running_step("3.2: stop_duration_box_plots")
        """Stop duration box plots"""
        # TODO: GTFS implementation should consider frequencies.txt.
        stop_duration_box_plots.plot_stop_duration_boxes(
            scenario_dir, plot_blotches=False, **kwargs)
    if 3 in steps or 3.3 in steps:
        print_running_step("3.3: temporal_clustering")
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

    if 4.1 in steps:
        print_running_step("4.1: routing")
        """Routing"""
        routing.build_routes(scenario_dir, **kwargs)
    if 4.2 in steps:
        print_running_step("4.2: fcd_conversion")
        """Skipping Routing (i.e. If the original data is already at a high
        frequency.)"""
        fcd_conversion.convert_data(scenario_dir, **kwargs)

    if 5.1 in steps or 5.2 in steps:
        routing_was_done, fcd_conversion_was_done = \
            _check_routing_status(scenario_dir)
    if 5.1 in steps:
        print_running_step("5.1: sumo_ev_simulation")
        """Simulation"""
        if not routing_was_done:
            error = "Please run the routing step before proceeding."
            LOGGERS['main'].error(error)
            raise ValueError(error)
        sumo_ev_simulation.simulate_all_routes(
            scenario_dir, skip_existing=False, **kwargs)
        # De-combine simulation results.
        results_splitter.split_results(scenario_dir, **kwargs)
    if 5.2 in steps:
        print_running_step("5.2: hull_ev_simulation")
        # TODO Remove this prompt, and rather make it automatically select
        # center integration.
        integration_mthd = None
        while integration_mthd is None:
            _ = auto_input("Would you like to do center, forward or backward integration? [center]/forward/backward  ", 'center', **kwargs)
            if _.lower() == 'center' or _ == '':
                integration_mthd = hull_ev_simulation.INTEGRATION_MTHD['ctr']
            elif _.lower() == 'forward':
                integration_mthd = hull_ev_simulation.INTEGRATION_MTHD['fwd']
            elif _.lower() == 'backward':
                integration_mthd = hull_ev_simulation.INTEGRATION_MTHD['bwd']
            else:
                print("Bad input!")

        if routing_was_done:
            sumo_ev_simulation.simulate_all_routes(
                scenario_dir, skip_existing=False, **kwargs)
            results_splitter.split_results(scenario_dir, **kwargs)
            # Backup SUMO's results, as they will be replaced by Hull's results.
            ev_out_dir = scenario_dir.joinpath('EV_Simulation',
                                               'EV_Simulation_Outputs')
            ev_out_dir.rename(ev_out_dir.parent.joinpath(
                              ev_out_dir.name + '.sumo.bak'))
            ev_out_dir.mkdir(parents=True, exist_ok=True)
            hull_ev_simulation.simulate(scenario_dir, integration_mthd,
                routing_was_done, **kwargs)
                # TODO  Remove integration_mthd argument.
        elif fcd_conversion_was_done:
            hull_ev_simulation.simulate(scenario_dir, integration_mthd,
                routing_was_done, **kwargs)
        else:
            raise ValueError("The Mobility Simulation step was not done. "
                "Please run it before requesting an EV Simulation.")

    if 6 in steps or 6.1 in steps:
        print_running_step("6.1: ev_results_analysis")
        """Generate Plots and Statistics from EV Simulation Results"""
        ev_results_analysis.run_ev_results_analysis(scenario_dir, **kwargs)
        ev_box_plots.plot_ev_energy_boxes(scenario_dir, **kwargs)
    if 6 in steps or 6.2 in steps:
        print_running_step("6.2: pv_results_analysis")
        """Generate Plots and Statistics from PV Simulation Results"""
        if kwargs['input_data_fmt'] == DATA_FMTS['GPS']:
            pv_results_analysis.run_pv_results_analysis(scenario_dir, **kwargs)
        elif kwargs['input_data_fmt'] == DATA_FMTS['GTFS']:
            print("Warning: PV Results Analysis is not implemented for GTFS " +
                  "scenarios yet.")
        else:
            raise ValueError(DATA_FMT_ERROR_MSG)
    if 6 in steps or 6.3 in steps:
        print_running_step("6.3: wind_results_analysis")
        """Generate Plots and Statistics from Wind Simulation Results"""
        if kwargs['input_data_fmt'] == DATA_FMTS['GPS']:
            wind_results_analysis.run_wind_results_analysis(scenario_dir, **kwargs)
        elif kwargs['input_data_fmt'] == DATA_FMTS['GTFS']:
            print("Warning: Wind Results Analysis is not implemented for GTFS" +
                  "scenarios yet.")
        else:
            raise ValueError(DATA_FMT_ERROR_MSG)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'scenario_dir', nargs='?', default=None, metavar='str',
        help='The root directory of the scenario to be simulated.')
    parser.add_argument(
        '--steps', default=None, metavar='List[float]',
        help="The steps to be run on the scenario as a comma-seperated list " +
             " of floats without spaces (e.g. '1,2.2,4').")
    parser.add_argument(
        '--debug', action='store_true',
        help="For developers: Starts the program with a breakpoint to help "
        "debugging, and also creates more detailed logging.")
    parser.add_argument(
        '--incl-weekends', action='store_true', help="Normally ev-fleet-sim "
        "discards weekend trips in the analysis (in step 2.2: Date filtering "
        "and seperation). Use this flag, to include weekend trips.")
    parser.add_argument(
        '--auto-run', action='store_true', help="Skip all prompts, "
        "automatically selecting the default values.")
    parser.add_argument(
        '-v', '--version', action='store_true',
        help="Print ev-fleet-sim's version, and quit.")
    args = parser.parse_args()

    if args.version:
        print(f"EV-Fleet-Sim version {version('ev-fleet-sim')}")
        sys.exit(0)

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
            if 5 in steps:
                step5 = float(input("You selected to run step 5. "
                    "Would you like to run 5.1 or 5.2?  "))
                steps.append(step5)
                if step5 == 5.2 and 4 in steps:
                    step4 = float(input("You selected to run step 4. "
                        "Would you like to run 4.1 or 4.2?  "))
                    steps.append(step4)
            if 4 in steps:
                step4 = float(input("You selected to run step 4. "
                    "Would you like to run 4.1 or 4.2?  "))
                steps.append(step4)
            if 5.1 in steps and 4.2 in steps:
                LOGGERS['main'].error("Steps 5.1 and 4.2 are not compatible!")
                raise ValueError("Steps 5.1 and 4.2 are not compatible!")
        except ValueError:
            LOGGERS['main'].error("Your steps selection was not valid! Try again...")
            time.sleep(1)
            continue
        break

    LOGGERS['main'].info(f"Running ev-fleet-sim with steps: \n\t{steps}")

    if args.auto_run:
        auto_run = True
    else:
        _ = input("Would you like to skip all prompts and use only default " +
                  "values? y/[n] ")
        auto_run = True if _.lower() == 'y' else False

    kwargs = {'auto_run': auto_run, 'input_data_fmt': None,
              'incl_weekends': args.incl_weekends,
              'debug_mode': args.debug}

    # Run all the steps
    _run(scenario_dir, steps, **kwargs)
