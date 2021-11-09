from area import area
from pathlib import Path
import pandas as pd


def get_map_size(scenario_dir: Path, **kwargs):
    print("Calculating map size...")
    boundary_file = [*scenario_dir.joinpath('_Inputs', 'Map',
                                            'Boundary').glob('*.csv')][0]
    boundary = pd.read_csv(boundary_file, skipinitialspace=True,
                           sep='\\s*,\\s*', engine='python')
    coordinates = [[list(vertex) for vertex in boundary.to_numpy()]]
    obj = {'type': 'Polygon', 'coordinates': coordinates}

    # Print the area in km^2
    print("Map size: {0:.2f} km^2".format(area(obj) / 10**6))
