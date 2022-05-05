"""Module to make scenario sub-directories if they don't exist."""

from pathlib import Path
import shutil
from typing import Dict
import json
import data_processing_ev as dpr
import platform


def initialise_scenario(scenario_dir: Path, **kwargs):
    """Initialise the scenario's folder structure."""

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

        # copy boundary.csv
        from_file = Path(__file__).parent.joinpath(
            'Initialisation_Files', 'boundary.csv')
        to_file = scenario_dir.joinpath(
            '_Inputs', 'Map', 'Boundary', 'boundary.csv')
        shutil.copy(from_file, to_file)

        # copy ./Initialisation_Files/Map_Construction/netconvert.sh
        from_file = Path(__file__).parent.joinpath(
            'Initialisation_Files', 'Map_Construction', 'netconvert.sh')
        to_file = scenario_dir.joinpath(
            '_Inputs', 'Map', 'Construction', 'netconvert.sh')
        shutil.copy(from_file, to_file)

        # copy ./Initialisation_Files/Map_Construction/pbf_to_osm.sh
        if platform.system() == 'Windows':
            from_file = Path(__file__).parent.joinpath(
                'Initialisation_Files', 'Map_Construction', 'pbf_to_osm.bat')
            to_file = scenario_dir.joinpath(
                '_Inputs', 'Map', 'Construction', 'pbf_to_osm.bat')
            shutil.copy(from_file, to_file)
        else:
            from_file = Path(__file__).parent.joinpath(
                'Initialisation_Files', 'Map_Construction', 'pbf_to_osm.sh')
            to_file = scenario_dir.joinpath(
                '_Inputs', 'Map', 'Construction', 'pbf_to_osm.sh')
            shutil.copy(from_file, to_file)

        # copy `./Initialisation_Files/Map_Construction/
        # osmNetconvert_Africa.typ.xml`
        from_file = Path(__file__).parent.joinpath(
            'Initialisation_Files', 'Map_Construction',
            'osmNetconvert_Africa.typ.xml')
        to_file = scenario_dir.joinpath(
            '_Inputs', 'Map', 'Construction', 'osmNetconvert_Africa.typ.xml')
        shutil.copy(from_file, to_file)

        # Copy `./initialisation-instructions.md`
        from_file = Path(__file__).parent.joinpath(
            'initialisation-instructions.md')
        to_file = scenario_dir.joinpath(
            'initialisation-instructions.md')
        shutil.copy(from_file, to_file)

    def create_readme():
        with open(str(scenario_dir.joinpath('directory_structure.json')), 'w') as f:
            json.dump(dpr.SCENARIO_DIR_STRUCTURE, f, indent=4)
        return

    create_dir(scenario_dir, dpr.SCENARIO_DIR_STRUCTURE)
    # Add files from ./Default_Files into the scenario
    copy_default_files()
    create_readme()
    print("Follow the initialisation instructions in: \n\t" +
          str(scenario_dir.joinpath("initialisation-instructions.md")) +
          "\n\tbefore preoceeding with the remaining simulation steps.")
