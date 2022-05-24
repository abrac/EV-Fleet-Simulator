#!/usr/bin/env python3

"""
This script takes a monolithic battery.out.xml file and splits it into multiple
battery.out.xml files in sub-directories, organised by ev_name and date.
"""


import os
import platform
import subprocess
import gc
import xml.etree.ElementTree as et
from pathlib import Path
from xml.dom import minidom
# from memory_profiler import profile
import multiprocessing as mp
from itertools import repeat
import data_processing_ev as dpr

from tqdm import tqdm

PIGZ_WARNING_ACKNOWLEDGED = False

def _split_ev_xml(ev_xml_file: Path, scenario_dir: Path,
                  input_data_fmt: int):

    file_stem = ev_xml_file.stem
    file_stem = file_stem[:file_stem.find('.out')]

    # Get the length fo the file.
    print("Getting the length of the xml file...")
    with open(ev_xml_file) as f:
        num_lines = sum(1 for line in f)

    tree = et.iterparse(ev_xml_file, events=("start", "end"))
    tree = iter(tree)
    _, root = tree.__next__()

    ev_name = ev_xml_file.parent.name
    print(f"### {ev_name} ###")

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
                # Save the tmp_root as an xml_file.
                ev_name = '_'.join(prev_id.split('_')[:-1])
                date = prev_id.split('_')[-1]
                output_file = scenario_dir.joinpath('SUMO_Simulation',
                                                    'Simulation_Outputs',
                                                    ev_name, date,
                                                    f'{file_stem}.out.xml')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_tree = et.ElementTree(tmp_root)
                with open(output_file, 'wb') as f:
                    # f.write(et.tostring(tmp_root))
                    tmp_tree.write(f, encoding='UTF-8')
                # Clear the temp root
                root.clear()

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
    output_file = scenario_dir.joinpath('SUMO_Simulation',
                                        'Simulation_Outputs', ev_name, date,
                                        f'{file_stem}.out.xml')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_tree = et.ElementTree(tmp_root)
    with open(output_file, 'wb') as f:
        # f.write(et.tostring(tmp_root))
        tmp_tree.write(f, encoding='UTF-8')
    # Clear the temp root
    root.clear()

    # Compress the combined XML file.

    print("Compressing combined XML file.")
    try:
        subprocess.run(['pigz', '-p', str(mp.cpu_count() - 2), '--quiet',
                       str(ev_xml_file.absolute())], check=True)
    except subprocess.CalledProcessError:
        print("Warning: Pigz failed to compress the xml file.")
    except OSError:
        print("Warning: You probably haven't installed `pigz`. Install " +
              "it if you want the script to automagically compress your " +
              "combined XML files after it has been split!")
        global PIGZ_WARNING_ACKNOWLEDGED
        if not PIGZ_WARNING_ACKNOWLEDGED:
            input("For now, press enter to ignore.")
            PIGZ_WARNING_ACKNOWLEDGED = True


def split_results(scenario_dir: Path, **kwargs):
    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])

    for file_stem in ('battery', 'fcd'):
        # Load xml as etree iterparse.
        xmls = sorted([*scenario_dir.joinpath(
            'SUMO_Simulation', 'Simulation_Outputs_Combined').glob(
            f'*/{file_stem}.out.xml')])

        args = zip(xmls, repeat(scenario_dir, len(xmls)),
                   repeat(input_data_fmt, len(xmls)))

        # with mp.Pool(mp.cpu_count() - 1) as p:
        #     p.starmap(_split_ev_xml, args)

        for arg_set in args:
            _split_ev_xml(*arg_set)
