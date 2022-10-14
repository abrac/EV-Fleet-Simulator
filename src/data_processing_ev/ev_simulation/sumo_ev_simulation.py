"""
Runs all simulations generated by the routing script.
"""

# TODO Compress sumo outputs with zlib.

import os
from pathlib import Path
import xml.etree.ElementTree as et  # Used for checking depart time in rou.xml
from xml.dom import minidom  # for writing sumocfg xml file
from tqdm import tqdm
from time import sleep
import subprocess
import data_processing_ev as dpr

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
    xml_file = scenario_dir.joinpath('EV_Simulation', 'SUMO_Simulation_Outputs',
                                     'battery.out.xml')
    xml_tree = et.parse(xml_file)
    # For each ev_name, date in route_files:
    # Find all `timestep` nodes with sub-node `vehicle` with attribute
    # `id`="{ev_name}_{date}"
    # TODO TODO FIXME FIXME XXX XXX


def simulate_all_routes(scenario_dir: Path, skip_existing: bool, **kwargs):

    # TODO Remove kurcheveil battery model if another custom battery model is
    # selected.

    auto_run = kwargs.get('auto_run', False)
    routes_dir = scenario_dir.joinpath('Routes', 'Routes')
    combined_route_files = sorted([*scenario_dir.joinpath(
        'Routes', 'Combined_Routes').glob('*.rou.xml')])

    if not auto_run:
        _ = input("Would you like to run each simulation in sumo-gui or sumo? " +
                  "sumo-gui/[sumo] ")
        gui = True if _.lower() == 'sumo-gui' else False
    else:
        gui = False

    # Check if any sumocfgs have already been generated.
    generate_sumocfgs = True
    output_sumocfg_dir = scenario_dir.joinpath('EV_Simulation', 'Sumocfgs_Combined')
    if [*output_sumocfg_dir.glob('*.sumocfg')]:
        _ = input("SUMO configuration files already found in \n\t" +
                  str(output_sumocfg_dir.absolute()) +
                  "\n\nWould you like to re-generate and overwrite the " +
                  "existing configuration files? [y]/n ")
        generate_sumocfgs = True if _.lower() != 'n' else False

    if generate_sumocfgs:
        for route_file in combined_route_files:
            ev_name = route_file.stem.split('.')[0]
            output_sumocfg_file = output_sumocfg_dir.joinpath(f'{ev_name}.sumocfg')
            simulation_output_dir = scenario_dir.joinpath(
                'EV_Simulation', 'SUMO_Simulation_Outputs_Combined', ev_name)
            output_sumocfg_dir.mkdir(parents=True, exist_ok=True)
            simulation_output_dir.mkdir(parents=True, exist_ok=True)

            # XML Creation
            # ------------

            configuration_node = et.Element('configuration')

            # TODO: Make these relative paths rather than absolute paths. See how I
            # did it for the `<output-prefix ... />` element for example.

            # inputs
            input_node = et.SubElement(configuration_node, 'input')
            et.SubElement(
                input_node,
                'net-file',
                {'value': os.path.relpath(
                    [*scenario_dir.joinpath('_Inputs', 'Map').\
                     glob('*.net.xml')][0].absolute(),
                    output_sumocfg_dir)})
            et.SubElement(input_node, 'additional-files', {
                'value': os.path.relpath(
                    routes_dir.joinpath('vType.add.xml'),
                    output_sumocfg_dir)})
            # Uncomment below if using multiple route files.
            # et.SubElement(input_node, 'route-files', {'value': route_str_list})
            et.SubElement(input_node, 'route-files', {
                'value': os.path.relpath(route_file, output_sumocfg_dir)}
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
                          {'value': 'battery.out.xml'})
            et.SubElement(output_node, 'fcd-output',
                          {'value': 'fcd.out.xml'})
            # TODO Prompt whether the user wants geo outputs or not.
            et.SubElement(output_node, 'fcd-output.geo',
                          {'value': 'true'})

            # processing
            processing_node = et.SubElement(configuration_node, 'processing')
            et.SubElement(processing_node, 'max-num-vehicles', {'value': '1'})
            # et.SubElement(processing_node, 'threads', {'value': f'{mp.cpu_count() -1}'})

            xmlstr = minidom.parseString(
                et.tostring(configuration_node)
            ).toprettyxml(indent="    ")
            with open(output_sumocfg_file, 'w') as f:
                f.write(xmlstr)

    simulation_cmds = []
    sumocfgs = sorted([*output_sumocfg_dir.glob('*.sumocfg')])
    for sumocfg_file in sumocfgs:
        # Run the simulation
        sumo_exe = "sumo-gui" if gui else "sumo"
        simulation_cmds.append([sumo_exe, '-c',
                                str(sumocfg_file.absolute()),
                                '--ignore-route-errors', 'true'])

    print('\n' * 3)

    print("Running simulations...")
    sleep(1)

    print("If simulations fail with \"Error: " +
          "basic_string::_M_replace_aux\", try increasing the limit of " +
          "maximum files open per process: `ucount -n 2048`")
    sleep(1)

    # INFO: This could be multi-threaded, but huge ammounts of ram would be
    #       needed.
    for simulation_cmd, sumocfg in tqdm(zip(simulation_cmds, sumocfgs)):
        ev_name = sumocfg.stem

        print(f"# Simulating {ev_name}...")
        subprocess.run(simulation_cmd)

        # Compress the simulation results.
        for file_stem in ('battery', 'fcd'):
            xml_file = scenario_dir.joinpath(
                'EV_Simulation', 'SUMO_Simulation_Outputs_Combined',
                ev_name, f'{file_stem}.out.xml')
            dpr.compress_file(xml_file)

    print("Simulations complete.")

    # Compressing simulation results
    print("Compressing simulation results.")



    # Seperating combined battery.out.xml file into serpate files, organised
    # by ev_name and date...
    # print("Seperating combined battery.out.xml file")
