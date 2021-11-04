# TODO Maybe convert the dir structure to format in
# https://stackoverflow.com/a/62570479/10462623 OR Create the dir structure in
# the file system, and store it using Path.glob('**')

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
from .ev_simulation import ev_simulation
from .ev_simulation import results_splitter
from .results_analysis import ev_results_analysis
from .results_analysis import ev_box_plots
from .results_analysis import pv_results_analysis

DATA_FMTS = {'GPS': 1, 'GTFS': 2}

DATA_FMT_ERROR_MSG = "The function has not been implemented for the " + \
                     "currently selected input data format."

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
    'SUMO_Simulation': {
        'Simulation_Outputs': None,
        'Sumocfgs': None
    },
    'SAM_Simulation': {
        'Results': None,
        'SAM_Scenario_File': None
    },
    'Results': {
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
    5.1. ev_simulation
    5.2. result_splitting
6. results_analysis
    6.1. ev_results_analysis
    6.2. pv_results_analysis
"""
