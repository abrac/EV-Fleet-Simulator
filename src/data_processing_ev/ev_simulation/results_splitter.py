#!/usr/bin/env python3

"""
This script takes a monolithic battery.out.xml file and splits it into multiple
battery.out.xml files in sub-directories, organised by ev_name and date.
"""


import os
import platform
import gc
import xml.etree.ElementTree as et
from pathlib import Path
from xml.dom import minidom
# from memory_profiler import profile
from itertools import repeat
import data_processing_ev as dpr
from tqdm import tqdm
from time import sleep
import subprocess
import sys
import typing as typ

if "SUMO_HOME" in os.environ:
    xml2csv = Path(os.environ["SUMO_HOME"], "tools", "xml", "xml2csv.py")
else:
    sys.exit("Please declare environmental variable 'SUMO_HOME'. It must " +
             "point to the root directory of the SUMO program.")


def _split_ev_xml(ev_xml_file: Path, scenario_dir: Path,
        input_data_fmt: int, skipping: bool):

    ev_name = ev_xml_file.parent.name
    print(f"### {ev_name} ###")

    output_dir = scenario_dir.joinpath(
        'EV_Simulation', 'EV_Simulation_Outputs', ev_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If there is something in the directory and you are skipping non-empty
    # directories, then exit. Otherwise, split the SUMO outputs.
    if any(output_dir.iterdir()) and skipping:
        print("Skipping {ev_name}, as it has already been split...")
        ev_xml_file = dpr.compress_file(ev_xml_file)
        return

    else:
        print("Decompressing the xml file...")
        ev_xml_file = dpr.decompress_file(ev_xml_file)

        file_stem = ev_xml_file.stem
        file_stem = file_stem[:file_stem.find('.out')]

        # Get the length fo the file.
        print("Getting the length of the xml file...")

        # Wait until the file has become *fully* decompressed.
        while not ev_xml_file.exists():
            pass
        file_written = False
        while not file_written:
            try:
                fp = open(ev_xml_file, 'a')
                fp.close()
                file_written = True
            except IOError:
                file_written = False

        with open(ev_xml_file, 'r') as f:
            num_lines = sum(1 for line in f)

        tree = et.iterparse(ev_xml_file, events=("start", "end"))
        tree = iter(tree)
        _, root = tree.__next__()

        print("Extracting the routes...")
        first_iteration = True
        # For each  `timestep` node in the iterator:
        route_count = 0
        time_offset_secs = 48 * 3600
        time_offset = route_count * time_offset_secs
        next_time_offset = (route_count + 1) * time_offset_secs - 1
        prev_id = None
        tmp_root = None
        for event, node in tqdm(tree, total=num_lines):

            # If the current node is a 'timestep' node, skip it. But keep a record
            #   of the time.
            if node.tag == 'timestep':
                time = int(float(node.get('time')))
                if time == next_time_offset:
                    route_count += 1
                    time_offset = route_count * time_offset_secs
                    next_time_offset = (route_count + 1) * time_offset_secs - 1
                continue

            # If the current node is a 'battery-/fcd-export' node, skip it.
            if node.tag == f'{file_stem}-export':
                continue

            if event == 'start':
                continue

            # Extract ev_name and date from id.
            id = node.get('id')

            if first_iteration or id != prev_id:

                if not first_iteration:
                    # breakpoint()  # XXX Test me, please. There is a delay after saving the file, before the next one starts...
                    # Save the tmp_root as an xml_file.
                    ev_name = '_'.join(prev_id.split('_')[:-1])
                    date = prev_id.split('_')[-1]
                    output_file = scenario_dir.joinpath('EV_Simulation',
                                                        'EV_Simulation_Outputs',
                                                        ev_name, date,
                                                        f'{file_stem}.out.xml')
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    tmp_tree = et.ElementTree(tmp_root)
                    with open(output_file, 'wb') as f:
                        # f.write(et.tostring(tmp_root))
                        tmp_tree.write(f, encoding='UTF-8')
                    # Clear the temp root
                    root.clear()
                    dpr.compress_file(output_file)

                tmp_root = et.Element(f'{file_stem}-export')
                # Append current node, if it's not the last node.
                if node is not None:
                    time_in_route = time - time_offset
                    time_node = et.Element('timestep', {'time': f'{time_in_route}'})
                    time_node.append(node)
                    tmp_root.append(time_node)
                    first_iteration = False
                else:
                    raise ValueError("Node is non-type. Maybe it is the end of " +
                                     "the xml iterator?")

            # else, if this ID is the same as the previous node
            else:
                time_in_route = time - time_offset
                time_node = et.Element('timestep', {'time': f'{time_in_route}'})
                time_node.append(node)
                tmp_root.append(time_node)

            # Append nodes to tmp_root, until we find a node with a different id.
            prev_id = id

        # After the loop has exited, save the last node:

        # Save the tmp_root as an xml_file.
        ev_name = '_'.join(prev_id.split('_')[:-1])
        date = prev_id.split('_')[-1]
        output_file = scenario_dir.joinpath('EV_Simulation',
                                            'EV_Simulation_Outputs', ev_name, date,
                                            f'{file_stem}.out.xml')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_tree = et.ElementTree(tmp_root)
        with open(output_file, 'wb') as f:
            # f.write(et.tostring(tmp_root))
            tmp_tree.write(f, encoding='UTF-8')
        # Clear the temp root
        root.clear()
        dpr.compress_file(output_file)

        # Compress the combined XML file.

        print("Compressing combined XML file.")
        ev_xml_file = dpr.compress_file(ev_xml_file)


def _create_csvs(scenario_dir, **kwargs):
    """Convert all battery.out.xml and fcd.out.xml files to csv files and
       save them.

    They will be saved in a corresponding folder in
    {Scenario_Dir}/EV_Simulation/EV_Simulation_Outputs/
    """

    ev_sim_dirs = sorted([*scenario_dir.joinpath(
        'EV_Simulation', 'EV_Simulation_Outputs').glob('*/*/')])

    fcd_sim_dirs = sorted([*scenario_dir.joinpath(
        'Mobility_Simulation', 'FCD_Data').glob('*/*/')])

    battery_csvs = sorted([*scenario_dir.joinpath('EV_Simulation',
            'EV_Simulation_Outputs').glob('*/*/battery.out.csv')])
    fcd_csvs = sorted([*scenario_dir.joinpath('Mobility_Simulation', 'FCD_Data').\
            glob('*/*/fcd.out.csv*')])

    if len(battery_csvs) == 0 or len(fcd_csvs) == 0:
        _ = dpr.auto_input("Would you like to convert all " +
            "battery.out.xml and fcd.out.xml " +
            "files to csv? [y]/n \n\t", 'y', **kwargs)
        convert = (True if _.lower() != 'n' else False)
    else:
        _ = dpr.auto_input("Would you like to re-convert all " +
            "battery.out.xml and fcd.out.xml " +
            "files to csv? [y]/n \n\t", 'y', **kwargs)
        convert = (True if _.lower() != 'n' else False)

    if convert:
        _ = dpr.auto_input(
            "Would you like to skip existing csv files? y/[n] \n\t", 'n',
            **kwargs)
        skipping = (True if _.lower() == 'y' else False)

        print("\nConverting xml files to csv...")

        for ev_sim_dir in tqdm(ev_sim_dirs):
            # Try create ev_csv if it doesn't exist
            battery_csv_gz = ev_sim_dir.joinpath('battery.out.csv.gz')
            battery_csv = ev_sim_dir.joinpath('battery.out.csv')
            battery_xml_gz = ev_sim_dir.joinpath('battery.out.xml.gz')
            battery_xml = dpr.decompress_file(battery_xml_gz)
            if not battery_xml.exists():
                continue
            if skipping and (battery_csv.exists() or battery_csv_gz.exists()):
                continue
            else:
                battery_csv.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(['python', xml2csv, '-s', ',',
                                '-o', battery_csv, battery_xml])
                # Warn if battery_csv *still* doesn't exist
                if not battery_csv.exists():
                    dpr.LOGGERS['main'].warning(
                        "Failed to create: \n\t" + str(battery_csv))
                else:
                    # If creating the battery_csv was succesful, compress
                    # the battery_xml and battery_csv files.
                    dpr.compress_file(battery_xml)
                    dpr.compress_file(battery_csv)

        for fcd_sim_dir in tqdm(fcd_sim_dirs):
            # Try create ev_csv if it doesn't exist
            fcd_csv_gz = fcd_sim_dir.joinpath('fcd.out.csv.gz')
            fcd_csv = fcd_sim_dir.joinpath('fcd.out.csv')
            fcd_xml_gz = fcd_sim_dir.joinpath('fcd.out.xml.gz')
            fcd_xml = dpr.decompress_file(fcd_xml_gz)
            if not fcd_xml.exists():
                continue
            if skipping and (fcd_csv.exists() or fcd_csv_gz.exists()):
                continue
            else:
                fcd_csv.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(['python', xml2csv, '-s', ',',
                                '-o', fcd_csv, fcd_xml])
                # Warn if fcd_csv *still* doesn't exist
                if not fcd_csv.exists():
                    dpr.LOGGERS['main'].warning(
                        "Failed to create: \n\t" + str(fcd_csv))
                else:
                    # If creating the fcd_csv was succesful, compress
                    # the fcd_xml and fcd_csv files.
                    dpr.compress_file(fcd_xml)
                    dpr.compress_file(fcd_csv)


def split_results(scenario_dir: Path, **kwargs):
    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])

    skip_splitting = False
    if any(scenario_dir.joinpath('EV_Simulation',
            'EV_Simulation_Outputs').iterdir()):
        _ = dpr.auto_input("Would you like to skip EVs that have already been "
                           "(partially) split? y/[n]", 'n', **kwargs)
        skip_splitting = False if _.lower() != 'y' else True

    # Load xml as etree iterparse.
    xmls = sorted([*scenario_dir.joinpath(
        'EV_Simulation', 'EV_Simulation_Outputs_Combined').glob(
        '*/*.out.xml*')])

    args = zip(xmls, repeat(scenario_dir, len(xmls)),
               repeat(input_data_fmt, len(xmls)),
               repeat(skip_splitting, len(xmls)))

    # with mp.Pool(mp.cpu_count() - 1) as p:
    #     p.starmap(_split_ev_xml, args)

    for arg_set in args:
        _split_ev_xml(*arg_set)

    # Moving the split FCD files to the mobility simulation folder.
    for fcd_file in scenario_dir.joinpath('EV_Simulation',
            'EV_Simulation_Outputs').glob('*/*/fcd.out.xml*'):
        new_location = scenario_dir.joinpath('Mobility_Simulation', 'FCD_Data',
            fcd_file.parents[1].name, fcd_file.parent.name, fcd_file.name)
        new_location.parent.mkdir(parents=True, exist_ok=True)
        fcd_file.rename(new_location)

    _create_csvs(scenario_dir, **kwargs)
