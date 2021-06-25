"""Module to make scenario sub-directories if they don't exist."""

from pathlib import Path
import shutil
from typing import Dict
import json


def initialise_scenario(scenario_dir: Path):
    """Initialise the scenario's folder structure."""

    from .. import SCENARIO_DIR_STRUCTURE

    def create_dir(scenario_dir: Path, sub_dirs: Dict):
            """Make scenario sub-directories if they don't exist."""
            # Make the scenario_dir.
            scenario_dir.mkdir(parents=True, exist_ok=True)
            # If there are no sub_dirs, quit the function.
            if sub_dirs is None:
                return
            # If there are sub_dirs, create each of them with *their* sub_dirs.
            for sub_dir, sub_sub_dirs in sub_dirs.items():
                sub_dir = scenario_dir.joinpath(sub_dir)
                create_dir(sub_dir, sub_sub_dirs)

    def copy_default_files():
        # copy ev_template
        from_file = Path(__file__).parent.joinpath('Initialisation_Files',
                                                   'ev_template.xml')
        to_file = scenario_dir.joinpath('_Inputs', 'Configs',
                                        'ev_template.xml')
        shutil.copy(from_file, to_file)
        # copy custom_osm_test.template.sumocfg
        from_file = Path(__file__).parent.joinpath(
            'Initialisation_Files', 'custom_osm_test.template.sumocfg')
        to_file = scenario_dir.joinpath(
            '_Inputs', 'Configs', 'custom_osm_test.template.sumocfg')
        shutil.copy(from_file, to_file)

    def create_readme():
        with open(str(scenario_dir.joinpath('readme.json')), 'w') as f:
            json.dump(SCENARIO_DIR_STRUCTURE, f, indent=4)
        return

    create_dir(scenario_dir, SCENARIO_DIR_STRUCTURE)
    # Add files from ./Default_Files into the scenario
    copy_default_files()
    create_readme()
