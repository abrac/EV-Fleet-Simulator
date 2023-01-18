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
        src_files = []
        dest_files = []

        # copy ev_template
        src_files.append(Path(__file__).parent.joinpath('Initialisation_Files',
                                                        'ev_template.xml'))
        dest_files.append(scenario_dir.joinpath('_Inputs', 'Configs',
                                                'ev_template.xml'))

        # copy custom_osm_test.template.sumocfg
        src_files.append(Path(__file__).parent.joinpath(
            'Initialisation_Files', 'custom_osm_test.template.sumocfg'))
        dest_files.append(scenario_dir.joinpath(
            '_Inputs', 'Configs', 'custom_osm_test.template.sumocfg'))

        # copy boundary.csv
        src_files.append(Path(__file__).parent.joinpath(
            'Initialisation_Files', 'boundary.csv'))
        dest_files.append(scenario_dir.joinpath(
            '_Inputs', 'Map', 'Boundary', 'boundary.csv'))

        # copy ./Initialisation_Files/Map_Construction/netconvert.sh
        src_files.append(Path(__file__).parent.joinpath(
            'Initialisation_Files', 'Map_Construction', 'netconvert.sh'))
        dest_files.append(scenario_dir.joinpath(
            '_Inputs', 'Map', 'Construction', 'netconvert.sh'))

        # copy ./Initialisation_Files/Map_Construction/pbf_to_osm.sh
        if platform.system() == 'Windows':
            src_files.append(Path(__file__).parent.joinpath(
                'Initialisation_Files', 'Map_Construction', 'pbf_to_osm.bat'))
            dest_files.append(scenario_dir.joinpath(
                '_Inputs', 'Map', 'Construction', 'pbf_to_osm.bat'))
        else:
            src_files.append(Path(__file__).parent.joinpath(
                'Initialisation_Files', 'Map_Construction', 'pbf_to_osm.sh'))
            dest_files.append(scenario_dir.joinpath(
                '_Inputs', 'Map', 'Construction', 'pbf_to_osm.sh'))

        # copy `./Initialisation_Files/Map_Construction/
        # osmNetconvert_Africa.typ.xml`
        src_files.append(Path(__file__).parent.joinpath(
            'Initialisation_Files', 'Map_Construction',
            'osmNetconvert_Africa.typ.xml'))
        dest_files.append(scenario_dir.joinpath(
            '_Inputs', 'Map', 'Construction', 'osmNetconvert_Africa.typ.xml'))

        # Copy `./initialisation-instructions.md`
        src_files.append(Path(__file__).parent.joinpath(
            'initialisation-instructions.md'))
        dest_files.append(scenario_dir.joinpath(
            'initialisation-instructions.md'))

        for src_file, dest_file in zip(src_files, dest_files):
            if not dest_file.exists():
                shutil.copy(src_file, dest_file)
            else:
                dpr.LOGGERS['main'].warning(
                    "Distination file already exists: \n\t\t " +
                    str(dest_file.absolute()))

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
