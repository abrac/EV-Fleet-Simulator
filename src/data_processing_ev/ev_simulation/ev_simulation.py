"""
Runs all simulations generated by the routing script.
"""

# TODO Compress sumo outputs with zlib.

import os
from pathlib import Path
import xml.etree.ElementTree as et  # Used for checking depart time in rou.xml
from xml.dom import minidom  # for writing sumocfg xml file
from tqdm import tqdm
import multiprocessing as mp  # for getting cpu count

# TODO This is deprecated
def get_ev_depart_time(sumocfg_file: Path) -> int:
    """Return EV's depart time in seconds.

    NB: Just looks for any rou.xml file in the sumocfg_file's path.
    """
    route_files = [*sumocfg_file.parent.glob('*.rou.xml')]
    if len(route_files) != 1:
        raise IndexError("More than one route file, or no route files found " +
                         "in sumocfg's directory.")
    route_xml = et.parse(route_files[0])
    try:
        depart_time = route_xml.getroot().find('vehicle').get('depart')
    except AttributeError as e:
        print(e)
        print("This error happened on file: \n\t" +
              str(sumocfg_file.absolute()))
    return int(float(depart_time))

def _seperate_battery_xml(scenario_dir: Path):
    xml_file = scenario_dir.joinpath('SUMO_Simulation', 'Simulation_Outputs',
                                     'Battery.out.xml')
    xml_tree = et.parse(xml_file)
    # For each ev_name, date in route_files:
    # Find all `timestep` nodes with sub-node `vehicle` with attribute
    # `id`="{ev_name}_{date}"
    # TODO TODO FIXME FIXME XXX XXX

def simulate_all_routes(scenario_dir: Path, skip_existing: bool):
    route_dir = scenario_dir.joinpath('Routes', 'Routes')
    route_files = sorted([*route_dir.glob('Combined/*.rou.xml')])
    for route_file in route_files:
        ev_name = route_file.stem.split('.')[0]
        output_sumocfg_dir = scenario_dir.joinpath('SUMO_Simulation', 'Sumocfgs', 'Combined')
        output_sumocfg_file = output_sumocfg_dir.joinpath(f'{ev_name}.sumocfg')
        simulation_output_dir = scenario_dir.joinpath(
            'SUMO_Simulation', 'Simulation_Outputs', 'Combined', ev_name)
        output_sumocfg_dir.mkdir(parents=True, exist_ok=True)
        simulation_output_dir.mkdir(parents=True, exist_ok=True)

        # XML Creation
        # ------------

        configuration_node = et.Element('configuration')

        # inputs
        input_node = et.SubElement(configuration_node, 'input')
        et.SubElement(
            input_node,
            'net-file',
            {
                'value': str(
                    [*scenario_dir.joinpath('_Inputs', 'Map').glob('*.net.xml')][0]
                )
            }
        )
        et.SubElement(input_node, 'additional-files', {
            'value': str(route_dir.joinpath('vType.add.xml').absolute())}
        )
        # Uncomment below if using multiple route files.
        # et.SubElement(input_node, 'route-files', {'value': route_str_list})
        et.SubElement(input_node, 'route-files', {
            'value': str(route_file.absolute())}
        )

        # outputs
        output_node = et.SubElement(configuration_node, 'output')
        et.SubElement(output_node, 'output-prefix',
                      {'value':
                       os.path.relpath(simulation_output_dir, output_sumocfg_dir) +
                       '/'})
        # et.SubElement(output_node, 'summary-output', {'value': 'summary.xml'})
        et.SubElement(output_node, 'tripinfo-output',
                      {'value': 'tripinfo.xml'})
        et.SubElement(output_node, 'battery-output',
                      {'value': 'Battery.out.xml'})

        # processing
        processing_node = et.SubElement(configuration_node, 'processing')
        et.SubElement(processing_node, 'max-num-vehicles', {'value': '1'})
        # et.SubElement(processing_node, 'threads', {'value': f'{mp.cpu_count() -1}'})

        xmlstr = minidom.parseString(
            et.tostring(configuration_node)
        ).toprettyxml(indent="    ")
        with open(output_sumocfg_file, 'w') as f:
            f.write(xmlstr)

        # Run the simulation
        print(f"Please run \n\t sumo -c \"{output_sumocfg_file.absolute()}\" " +
              "--ignore-route-errors true")

    print("If simulations fail with \"Error: basic_string::_M_replace_aux\", try " +
          "increasing the limit of maximum files open per process: " +
          "`ucount -n 2048`")
    input("Press enter once the sumo simulation has been completed...")

    # Seperating combined Battery.out.xml file into serpate files, organised
    # by ev_name and date...
    print("Seperating combined Battery.out.xml file")

    breakpoint()  # XXX
    _seperate_battery_xml(scenario_dir)

    # TODO Please, fix: This is giving the following error:

    # Traceback (most recent call last):
    #   File "/home/c_abraham/Documents/Git_Repositories/etaxi/Experiments/data_processing_ev/./main.py", line 86, in <module>
    #   File "/home/c_abraham/Documents/Git_Repositories/etaxi/Experiments/data_processing_ev/./main.py", line 62, in run
    #   File "/home/c_abraham/Documents/Git_Repositories/etaxi/Experiments/data_processing_ev/data_processing_ev/ev_simulation/ev_simulation.py", line 96, in simulate_all_routes
    #   File "/usr/share/sumo/tools/libsumo/__init__.py", line 153, in start
    #   File "/usr/share/sumo/tools/libsumo/libsumo.py", line 3259, in load
    # RuntimeError: unknown exception

    # libsumo.start(["sumo", "-c", str(output_sumocfg_file)])
    # while libsumo.simulation.getMinExpectedNumber() != 0:
    #     libsumo.simulationStep()
    # libsumo.close()  # waits until Sumo process is finished, then closes.


    # split the results of the simulation:
    # TODO fix the below
    # date = route_file.name.split('.')[0]
    # output_subdir = output_dir.joinpath(date)
    # if skip_existing and output_dir.joinpath('Battery.out.xml').exists():
    #     continue

    # FIXME Below is legacy stuff
    # for sumocfg_file in tqdm(sumocfg_files):
    #     if (skip_existing and
    #             sumocfg_file.parent.joinpath('Battery.out.xml').exists()):
    #         print("Skipping...")
    #         continue