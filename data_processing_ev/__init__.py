# TODO Maybe convert the dir structure to format in
# https://stackoverflow.com/a/62570479/10462623 OR Create the dir structure in
# the file system, and store it using Path.glob('**')

from .scenario_initialisation.scenario_initialisation \
    import initialise_scenario
from . import data_visualisation  # TODO Make this consistent with the others.
from .spatial_clustering import spatial_clustering
from .spatial_clustering import spatial_filtering
from .temporal_clustering import temporal_clustering
from .routing import routing
from .ev_simulation import ev_simulation
from .results_analysis import results_analysis

SCENARIO_DIR_STRUCTURE = {
    '_Inputs': {
        'Traces': {
            'Original': None,
            'Processed': None
        },
        'Map': {
            'Boundary': None
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
        'Graphs': None
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
        'EV_Energy_Usage': None,
        'EV_Stop_Durations': None,
        'PV_Charging_Potential': None
    }
}

MODULES = """
0. scenario_initialisation
1. data_visualisation
    1. mapping
    2. route_animation
2. spatial clustering and filtering
    1. spatial_clustering
    2. date_filtering_and_separation
3. temporal_clustering (and filtering)
4. routing
5. simulation
6. results_generation
"""
