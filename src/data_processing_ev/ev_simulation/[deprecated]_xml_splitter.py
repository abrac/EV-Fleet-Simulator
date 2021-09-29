#!/usr/bin/env python3

"""
This script takes a monolithic battery.out.xml file and splits it into multiple
battery.out.xml files in sub-directories, organised by ev_name and date.
"""


import os
import gc
import xml.etree.ElementTree as et
from pathlib import Path
from xml.dom import minidom
from memory_profiler import profile

from tqdm import tqdm


def main(scenario_dir: Path, **kwargs):

    # Load xml as etree iterparse.
    monolithic_xml = scenario_dir.joinpath('SUMO_Simulation',
                                           'Simulation_Outputs',
                                           'Battery.out.xml')
    tree = et.iterparse(monolithic_xml, events=("start", "end"))
    tree = iter(tree)
    _, root = tree.__next__()

    # with open(monolithic_xml) as f:
    #     num_lines = sum(1 for line in f)
    num_lines = 129835700  # Get this number using `wc -l Battery.out.xml`.
    #   Only works on Linux. Uncomment the above code if on Windows.

    first_iteration = True
    # For each  `timestep` node in the iterator:
    route_count = 0; time_offset_secs = 48*3600
    time_offset = route_count * time_offset_secs
    next_time_offset = (route_count + 1) * time_offset_secs - 1
    for event, node in tqdm(tree, total=num_lines*6.5):

        # If the current node is a 'timestep' node, skip it. But keep a record
        #   of the time.
        if node.tag == 'timestep':
            time = int(float(node.get('time')))
            if time == next_time_offset:
                route_count += 1
                time_offset = route_count * time_offset_secs
                next_time_offset = (route_count + 1) * time_offset_secs - 1
            continue

        if event == 'start':
            continue

        # Extract ev_name and date from id.
        id = node.get('id')

        if first_iteration or id != prev_id:

            if not first_iteration:
                # Save the tmp_root as an xml_file.
                ev_name = prev_id.split('_')[0]
                date = '_'.join(prev_id.split('_')[1:])
                output_file = scenario_dir.joinpath('SUMO_Simulation',
                                                    'Simulation_Outputs',
                                                    ev_name, date,
                                                    'Battery.out.xml')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_tree = et.ElementTree(tmp_root)
                with open(output_file, 'wb') as f:
                    #f.write(et.tostring(tmp_root))
                    tmp_tree.write(f, encoding='UTF-8')
                # Clear the temp root
                root.clear()

            tmp_root = et.Element('battery-export')
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


if __name__ == "__main__":
    scenario_dir = Path(os.path.abspath(__file__)).parents[2]
    main(scenario_dir)
