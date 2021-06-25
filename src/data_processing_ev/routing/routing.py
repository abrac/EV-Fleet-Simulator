#!/usr/bin/env python3

"""
Script to generate route file from csv gps traces.

Inputs:
    - Clean GPS traces, clustered and filtered. (CSV)
    - Template with EV Parameters to apply on the SUMO vehicle. (XML)
Outputs:
    - Sumo route files for each EV. (XML)

Author: Chris Abraham
"""

import os
import sys
import xml.etree.ElementTree as et
import logging
from xml.dom import minidom
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from operator import itemgetter
from itertools import repeat
from multiprocessing import Pool, cpu_count

# Import sumolib for locating nearby lanes from geo-coordinates
# Import duaiterate for routing between spatial clusters
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    assign_tools = os.path.join(os.environ['SUMO_HOME'], 'tools', 'assign')
    sys.path.append(assign_tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import sumolib  # noqa: Must be imported after adding tools to Path.
# import duaIterate


def generate_xml(net: sumolib.net.Net, df: pd.DataFrame, xml_template: Path,
                 vehicle_id: str) -> et.ElementTree:
    """
    Generate xml.

    Parameters:
    df (DataFrame): List of taxi stop coordinates in order
    xml_template (PosixPath): Posix path of xml template
    vehicle_id (String): ID of vehicle
    export (Bool): True if exporting xml as file
    xml_out (PosixPath): Posix path of file to export xml
    """
    # TODO Make these function arguments
    LANE_LOC_RADIUS_WARN = 50  # unit: meters. If distance is exceeded,
    #                            a warning will be logged.

    def seconds_since_midnight(time: str) -> int:
        """
        Return time in seconds after midnight in SUMO time format.

        Parameters:
        Time (String): Time in format "2013/12/02 10:34:29 AM"

        Returns:
        Int: Time in seconds
        """
        (_, time) = time.split(" ")
        (hr24, minute, sec) = time.split(":")
        return int(hr24) * 60 * 60 + int(minute) * 60 + int(sec)

        # (_, time, meridiem) = time.split(" ")
        # (hr12, minute, sec) = time.split(":")
        # hr24 = (int(hr12) % 12) + (12 if meridiem == "PM" else 0)
        # return hr24*60*60 + int(minute)*60 + int(sec)

    def lonlat2lane(lon: float, lat: float, net: sumolib.net,
                    vClass: str = None) -> sumolib.net.lane.Lane:
        """Return the lane which is closest to gps-coordinate <lon, lat>

        Parameters:
            vClass: if vClass is given, the function will return an lane
            that allows the vClass.
        """
        if vClass and (vClass not in sumolib.net.lane.SUMO_VEHICLE_CLASSES):
            raise ValueError("The vClass given is not a real SUMO vClass.")

        # retrieve lanes near <lon, lat>
        x, y = net.convertLonLat2XY(lon, lat)

        lane_loc_radius = 10  # unit: meters
        closest_lane = None
        while not closest_lane:
            dists_and_lanes = net.getNeighboringLanes(x, y, lane_loc_radius)
            if dists_and_lanes:
                # sort the lanes by ascending distance
                dists_and_lanes = [(dist, lane) for lane, dist in dists_and_lanes]
                dists_and_lanes = sorted(dists_and_lanes, key=itemgetter(0))
                # select the closest lane
                _, closest_lane = dists_and_lanes[0]

                # only keep lanes that allow the vClass
                dists_and_lanes_of_vclass = [
                    (dist, lane) for (dist, lane) in dists_and_lanes if
                    lane.allows(vClass)
                ]

                if dists_and_lanes_of_vclass:
                    closest_dist = dists_and_lanes_of_vclass[0][0]
                    closest_lane = dists_and_lanes_of_vclass[0][1]
                    if closest_dist > LANE_LOC_RADIUS_WARN:
                        logging.warn(f"""
                            Closest lane to <{lon}, {lat}> exceeds the warning
                            threshold: {LANE_LOC_RADIUS_WARN}.")
                        """)
            lane_loc_radius *= 2
        return closest_lane

    # Create an xml tree from the template in this file's path.
    tree = et.parse(xml_template)
    root = tree.getroot()
    vClass = root.find('vType').get('vClass')

    # Read the df of taxi-stop coordinates and append stops to xml tree.

    # Using a trip definition with stops
    if len(df):  # If dataframe is not empty
        # Create a trip from the first stop to the last stop
        stop_1 = df.iloc[0]
        stop_n = df.iloc[-1]
        stop_1_lane = lonlat2lane(stop_1.Longitude, stop_1.Latitude,
                                  net, vClass)
        stop_n_lane = lonlat2lane(stop_n.Longitude, stop_n.Latitude,
                                  net, vClass)
        trip_node = et.Element(
            "trip", {
                "id": vehicle_id,
                "type": "e_minibus_taxi",
                "depart": str(seconds_since_midnight(stop_1.Time)),
                "from": stop_1_lane.getEdge().getID(),
                "to": stop_n_lane.getEdge().getID()
            }
        )
        for idx, row in df.iloc[1:-1].iterrows():
            closest_lane = lonlat2lane(row.Longitude, row.Latitude,
                                       net, vClass)
            stop_node = et.Element(
                'stop', {
                    'lane': closest_lane.getID(),
                    'until': str(seconds_since_midnight(row.Time)),
                    'parking': 'true',
                    'friendlyPos': 'true'
                }
            )
            trip_node.append(stop_node)
        root.append(trip_node)

    return tree

# TODO Create file with statistics regarding gps coordinates which failed to
# find a nearby edge.


def _do_trip_building(input_file: Path, output_dir: Path, xml_template: Path,
                      skip_existing: bool, net: sumolib.net.Net):
    """Trip building multiprocessing function"""
    try:
        input_data = pd.read_csv(input_file)
    except pd.errors.EmptyDataError:
        # TODO Use the logging library for the below
        print("No data in file. Skipping...")
        log_file = output_dir.joinpath("log.txt")
        with open(log_file, 'a+') as f:
            f.write("Skipped file: " + input_file.name + "\n")
        return

    # Create a subpath in the output directory with the name of the current
    # taxi.
    #   TODO: Get the name using some other method. Rather than using the
    #   parent directory, use some way to store metadata.
    output_subdir = output_dir.joinpath(input_file.parent.name)
    output_subdir.mkdir(parents=True, exist_ok=True)
    export_file = output_subdir.joinpath(f"{input_file.stem}.trip.xml")
    if skip_existing and export_file.exists():
        # TODO Use logging library for the below
        print(f"{export_file.stem}.trip.xml exists. Skipping...")
        return

    tree = generate_xml(net, input_data, xml_template, input_file.stem)
    # export the xml tree as a file, using minidom to create a "pretty" xml
    # from our element tree
    #   TODO Remove the random spaces. (in the generated xml file)
    xmlstr = minidom.parseString(
        et.tostring(tree.getroot())).toprettyxml(indent="    ")
    with open(export_file, 'w') as f:
        f.write(xmlstr)


# TODO TODO TODO Switch to duaiterate or duarouter
def _do_route_building(trip_file: Path, output_dir: Path, skip_existing: bool,
                       net: sumolib.net.Net):
    """Route building multiprocessing function"""
    # Create a subpath in the output directory with the name of the current
    # taxi.
    #   TODO: Get the name and date using some other method. Rather than using
    #   the parent directory, use some way to store metadata.
    ev_name = trip_file.parent.name
    date = '{0}_{1}_{2}'.format(
        *trip_file.stem.split('.')[0].split('_')[1].split('-'))
    # output_subdir = output_dir.joinpath(ev_name, date)
    output_subdir = output_dir.joinpath(ev_name)
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_file = output_subdir.joinpath(f"{date}.rou.xml")
    # Choose between `duaiterate` (`not routing_locally`) and
    # `sumolib.net.Net.getShortestPath` (`routing_locally`)

    # Don't do any processing if we're skipping existing files and the rou.xml
    # file exists.
    if skip_existing and output_file.exists():
        return

    # ROUTING
    # -------

    # Import the stops from the trip.xml file as xml.etree
    trip_tree = et.parse(trip_file)
    trip_tree_root = trip_tree.getroot()
    vType_node = trip_tree_root.find('vType')
    # Create a list of pairs of stops [(s1, s2), (s2, s3), ...]
    trip_node = trip_tree_root.find('trip')
    first_stop = trip_node.attrib['from']
    last_stop = trip_node.attrib['to']
    middle_stops = [*trip_node.iter('stop')]
    middle_stop_edges = [
        middle_stop.attrib['lane'].split('_')[0] for middle_stop in
        middle_stops
    ]
    stop_pairs = []
    prev_stop = first_stop
    for middle_stop in middle_stop_edges:
        stop_pairs.append((prev_stop, middle_stop))
        prev_stop = middle_stop
    stop_pairs.append((prev_stop, last_stop))

    # Generate routes between each pair of stops and chain them
    vClass = vType_node.attrib['vClass']
    route = []
    route_edges = None
    for stop_pair in stop_pairs:
        route_edges, cost = net.getShortestPath(
            fromEdge=net.getEdge(stop_pair[0]),
            toEdge=net.getEdge(stop_pair[1]),
            vClass=vClass
        )
        # if a route was found between the stop_pair, append those edges to
        # `route`. Else, log an error.
        if route_edges:
            for route_edge in route_edges[:-1]:
                route.append(route_edge.getID())
        else:
            logging.error(f"{ev_name}: {date}: Failed to find a route " +
                          f"between edges {stop_pair[0]} and {stop_pair[1]}")
            # Assign the last edge of the stop_pair to `route edges`, *just in-
            # case* this is the last iteration of the loop. If `route_edges` is
            # not assigned, `route_edges[-1].getID()` (see a few lines below)
            # will throw an error.
            route_edges = [net.getEdge(stop_pair[-1])]
    # if chained route is empty, return. Don't continue saving the xml file.
    if len(route) < 1:
        logging.error(f"{ev_name} has no edges in its route on {date}!")
        return
    route.append(route_edges[-1].getID())
    route_str = " ".join(str(edge) for edge in route)
    route_node = et.Element('route', {'edges': route_str})

    # Construct rou.xml file as xml.etree
    route_tree_root = et.Element('routes')
    vehicle_node = et.SubElement(
        route_tree_root, 'vehicle', {
            'id': trip_node.attrib['id'],
            'type': trip_node.attrib['type'],
            'depart': trip_node.attrib['depart']
        }
    )
    vehicle_node.append(route_node)
    for middle_stop in middle_stops:
        vehicle_node.append(middle_stop)

    # TODO Maybe return the xml_tree and save in the outer function?
    # (to try get multiprocessing to work.)
    # Save the routes as a rou.xml file
    xmlstr = minidom.parseString(
        et.tostring(route_tree_root)
    ).toprettyxml(indent="    ")
    with open(output_file, 'w') as f:
        f.write(xmlstr)

    # Create additionals file with vType definition, if it doesn't exist
    # already.
    vtype_file = output_dir.joinpath('vType.add.xml')
    if not vtype_file.exists():
        vtype_tree_root = et.Element('additional')
        vtype_tree_root.append(vType_node)
        # TODO Maybe return the xml_tree and save in the outer function?
        # (to try get multiprocessing to work.)
        # Save the routes as a rou.xml file
        xmlstr = minidom.parseString(
            et.tostring(vtype_tree_root)
        ).toprettyxml(indent="    ")
        with open(vtype_file, 'w') as f:
            f.write(xmlstr)

    # FIXME Abandon this approach
    # else:
    #     # TODO Fix **critical** error: """
    #     #       FileNotFoundError: [Errno 2] No such file or directory:
    #     #       'main.py'
    #     #   """
    #     #   at `with io.open_code(filename) as fp:`.
    #     #   and """
    #     #       RecursionError: maximum recursion depth exceeded [of fp.write()]
    #     #   """

    #     # Store current_dir
    #     cwd = os.getcwd()
    #     # Change directory to output_subdir
    #     os.chdir(str(output_subdir.absolute()))
    #     try:
    #         duaIterate.main(
    #             [
    #                 # FIXME don't reimport the network each and every time!
    #                 '-n', str([
    #                     *scenario_dir.joinpath('_Inputs', 'Map').glob('*.net.xml')
    #                 ][0].absolute()),
    #                 '-t', str(trip_file.absolute()),
    #                 '-l', '1',
    #                 'duarouter--keep-vtype-distributions', 'true',
    #                 'duarouter--routing-threads', '1',
    #                 'sumo--threads', '1'
    #             ]
    #         )
    #     except SystemExit:
    #         logging.error("DuaIterate failed at: \n\t" +
    #                       str(output_subdir.absolute()))
    #     # Restore current_dir
    #     os.chdir(cwd)


def build_routes(scenario_dir: Path):
    # Configure logging
    loggingFile = scenario_dir.joinpath('Routes', 'generation_log.txt')
    logging.basicConfig(filename=loggingFile,  # encoding='utf-8',
                        level=logging.WARN)

    cluster_dir = scenario_dir.joinpath('Spatial_Clusters')
    input_list = sorted(
        [*cluster_dir.joinpath("Filtered_Traces").glob("*/*.csv")]
    )
    output_dir = scenario_dir.joinpath("Routes", "Trips")
    xml_template = scenario_dir.joinpath('_Inputs', 'Configs',
                                         'ev_template.xml')

    # Ask user if s/he wants to skip existing trip.xml files
    # TODO Ask this in main.py and only if `configuring` = True
    _ = input("Would you like to skip existing trip.xml files? y/[n]")
    skip_existing = True if _.lower() == 'y' else False

    print("Loading sumo network...")
    network_file = [
        *scenario_dir.joinpath('_Inputs', 'Map').glob('*.net.xml')
    ][0]
    net = sumolib.net.readNet(network_file)
    print("Done loading network.\n")

    print("Generating trip files...")

    # OLD MULTITHREADED WAY:
    # ----------------------
    # """ Doesn't work because it has too much recursive depth. """
    # args_trip = zip(
    #     input_list,
    #     repeat(output_dir, len(input_list)),
    #     repeat(xml_template, len(input_list)),
    #     repeat(skip_existing, len(input_list)),
    #     repeat(net, len(input_list)))
    # with Pool(cpu_count() - 3) as p:
    #     p.starmap(_do_trip_building, args_trip)

    # Do trip generation
    for input_file in tqdm(input_list):
        _do_trip_building(input_file, output_dir, xml_template,
                          skip_existing, net)
    print("Done generating trip files.")

    # Ask user if s/he wants to skip existing rou.xml files
    # TODO Ask this in main.py and only if `configuring` = True
    _ = input("Would you like to skip existing rou.xml files? y/[n]")
    skip_existing = True if _.lower() == 'y' else False

    print("Generating route files...")
    # Do route generation
    trip_files = [*output_dir.glob('*/*.trip.xml')]
    output_dir = scenario_dir.joinpath("Routes", "Routes")

    # OLD MULTITHREADED WAY:
    # ----------------------
    # """ Doesn't work because it has too much recursive depth. """
    # args_route = zip(
    #     trip_files,
    #     repeat(output_dir, len(input_list)),
    #     repeat(skip_existing, len(input_list)),
    #     repeat(net, len(input_list)))
    # with Pool(cpu_count() - 3) as p:
    #     p.starmap(_do_route_building, args_route)

    # IF DEBUGGING
    # ------------
    for trip_file in tqdm(trip_files):
        _do_route_building(trip_file, output_dir, skip_existing, net)
    print("Done generating route files.")

    # Ask user if s/he wants to skip combinding rou.xml files
    # TODO Ask this in main.py and only if `configuring` = True
    _ = input("Would you like to skip combining the routes? y/[n]")
    skip_combining_routes = True if _.lower() == 'y' else False

    # TODO Combine the routes on a per-taxi basis. Therefore, if there are 10
    # taxis, and I have 4 cpu threads, I can run 4 sumo simulations in parallel.
    if not skip_combining_routes:
        print("Combining routes into one file per EV.")
        # Combine all routes into one file per EV!
        ev_dirs = scenario_dir.joinpath("Routes", "Routes").glob('T*/')
        output_dir = scenario_dir.joinpath("Routes", "Routes", "Combined")
        output_dir.mkdir(parents=True, exist_ok=True)
        for ev_dir in tqdm(ev_dirs):
            route_files = [*ev_dir.glob('*.rou.xml')]
            route_files.sort()
            combined_rou_xml_file = output_dir.joinpath(
                f'{ev_dir.stem}.rou.xml')
            combined_rou_xml_root = et.Element('routes')
            for idx, route_file in tqdm(enumerate(route_files)):
                # Time offset to space out the beginning of each route by 48 hours.
                time_offset_hrs = 48
                time_offset = idx * time_offset_hrs * 3600
                route_xml_tree = et.parse(route_file)
                route_xml_root = route_xml_tree.getroot()

                #  Apply the time offset to the route `depart` time and `stop` `until`
                # times.

                vehicle_node = route_xml_root.find('vehicle')
                old_depart = int(vehicle_node.get('depart'))
                vehicle_node.set('depart', str(old_depart + time_offset))

                # For each `stop` in vehicle_node, add the time_offset to its `until`:
                stop_nodes = vehicle_node.findall('stop')
                for stop_node in stop_nodes:
                    old_until = int(stop_node.get('until'))
                    stop_node.set('until', str(old_until + time_offset))

                combined_rou_xml_root.append(vehicle_node)

            # Sort results by departure time. # FIXME Delete below
            # combined_rou_xml_root[:] = sorted(
            #     combined_rou_xml_root, key=lambda element: int((element.get('depart')))
            # )

            # Export the xml tree as a file, in a way that looks pretty.
            xml_bytestream = et.tostring(combined_rou_xml_root)
            with open(combined_rou_xml_file, 'wb') as f:
                f.write(xml_bytestream)
            print(f"Done combining routes for {ev_dir.stem}.")
