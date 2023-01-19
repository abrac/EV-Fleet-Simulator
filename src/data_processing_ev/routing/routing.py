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
import typing as t
from xml.dom import minidom
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from operator import itemgetter
from itertools import repeat
from multiprocessing import Pool, cpu_count
import data_processing_ev as dpr

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


def generate_route_xml(net: sumolib.net.Net, df: pd.DataFrame, xml_template: Path,
                 vehicle_id: str, stop_labels: t.List[bool]) -> et.ElementTree:
    """
    Generate a route xml file using the road network and the sequence of
    recorded stop events.

    Parameters:
    net (sumolib.net.Net): The SUMO network of the location of study.
    df (DataFrame): List of taxi geo-coordinates in order (as read from
                    Filtered traces)
    xml_template (PosixPath): Posix path of xml template
    vehicle_id (String): ID of vehicle
    export (Bool): True if exporting xml as file
    xml_out (PosixPath): Posix path of file to export xml
    """
    # TODO Make these function arguments
    LANE_LOC_RADIUS_WARN = 100  # unit: meters. If distance is exceeded,
                                # a warning will be logged.

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

        lane_loc_radius = 10  # unit: meters. Initialising lane_loc_radius.
            # It is automatically increased, while a lane is not found.
        closest_lane = None
        while not closest_lane:
            dists_and_lanes = net.getNeighboringLanes(x, y, lane_loc_radius)
            if dists_and_lanes:
                # sort the lanes by ascending distance
                dists_and_lanes = [(dist, lane) for lane, dist in dists_and_lanes]
                dists_and_lanes = sorted(dists_and_lanes, key=itemgetter(0))

                # only keep lanes that allow the vClass
                dists_and_lanes_of_vclass = [
                    (dist, lane) for (dist, lane) in dists_and_lanes if
                    lane.allows(vClass)
                ]

                if dists_and_lanes_of_vclass:
                    closest_dist = dists_and_lanes_of_vclass[0][0]
                    closest_lane = dists_and_lanes_of_vclass[0][1]
                    if closest_dist > LANE_LOC_RADIUS_WARN:
                        dpr.LOGGERS['main'].warn(f"""
                            Closest lane to <{lon}, {lat}> exceeds the warning
                            threshold: {LANE_LOC_RADIUS_WARN}.")
                        """)
            lane_loc_radius *= 2
        return closest_lane

    # Date
    date = df.iloc[0]['Time'].split(' ')[0]

    # Create an xml tree from the template, to get the vClass.
    template_tree = et.parse(xml_template)
    vClass = template_tree.getroot().find('vType').get('vClass')

    # Create an xml tree for the route file.
    root = et.Element('routes')
    tree = et.ElementTree(root)

    # Read the df of taxi-stop coordinates and append stops to xml tree.

    # Using a route definition with stops
    if len(df):  # If dataframe is not empty
        # Create a trip from the first stop to the last stop
        stop_1 = df.iloc[0]
        vehicle_node = et.Element(
            "vehicle", {
                "id": vehicle_id,
                "type": "e_minibus_taxi",
                "depart": str(seconds_since_midnight(stop_1.Time)),
            }
        )

        # A list of all the waypoints along the route.
        route_waypoints = []

        # A list of all the stop nodes along the route.
        stop_nodes = []

        for (_, row), (_, stop_label) in zip(df.iterrows(),
                                             stop_labels.iterrows()):
            stop_label = bool(stop_label['Stopped'])
            # Get the closest lane.
            closest_lane = lonlat2lane(row.Longitude, row.Latitude,
                                       net, vClass)

            # Check if the row is a stop or waypoint, by reading from the
            # `stop_labels_{ev_name}.csv` file.

            # If it is a stop:
            if stop_label is True:

                # Create a stop node.
                stop_node = et.Element(
                    'stop', {
                        'lane': closest_lane.getID(),
                        'until': str(seconds_since_midnight(row.Time)),
                        'parking': 'true',
                        'friendlyPos': 'true'
                    }
                )
                stop_nodes.append(stop_node)

            route_waypoints.append(closest_lane.getEdge().getID())

        # Construct route waypoint pairs [(edge1, edge2), (edge2, edge3), ...]
        waypoint_pairs = []
        prev_waypoint = route_waypoints[0]
        for waypoint in route_waypoints[1:]:
            waypoint_pairs.append((prev_waypoint, waypoint))
            prev_waypoint = waypoint

        # Generate routes between each pair of waypoints and chain them
        route = []
        last_found_edge = None
        for waypoint_pair in waypoint_pairs:
            # NOTE: I use the shortest path between waypoints. This is a
            # simplification.
            route_edges, cost = net.getShortestPath(
                fromEdge=net.getEdge(waypoint_pair[0]),
                toEdge=net.getEdge(waypoint_pair[1]),
                vClass=vClass
            )
            # if a route was found between the stop_pair, append those edges to
            # `route`. Else, log an error.
            if route_edges:
                for route_edge in route_edges[:-1]:
                    route.append(route_edge.getID())
                last_found_edge = route_edges[-1]
            else:
                dpr.LOGGERS['main'].error(
                    f"{vehicle_id}: {date}: Failed to find a route "
                    f"between edges {waypoint_pair[0]} and {waypoint_pair[1]}")

        # TODO: Below, is an alternative method to the above. It is better,
        # since it handles better the case when a route is not found, since it
        # always routes against the last *successful* route edge. However, if
        # the last route edge is in a nasty corner of the map, it can be
        # difficult for any subsequent waypoints to find a route from there.
        # Therefore, I am disabling for now.

        # last_route_edge = route_waypoints[0]
        # for route_waypoint in route_waypoints:
        #     # NOTE: I use the shortest path between waypoints. This is a
        #     # simplification.
        #     route_edges, cost = net.getShortestPath(
        #         fromEdge=net.getEdge(last_route_edge),
        #         toEdge=net.getEdge(route_waypoint),
        #         vClass=vClass
        #     )
        #     # if a route was found between the stop_pair, append those edges to
        #     # `route`. Else, log an error.
        #     if route_edges:
        #         for route_edge in route_edges[:-1]:
        #             route.append(route_edge.getID())
        #         last_route_edge = route_waypoint
        #     else:
        #         dpr.LOGGERS['main'].error(f"{vehicle_id}: {date}: Failed to find a route " +
        #                      f"between edges {last_route_edge} and {route_waypoint}")

        # if chained route is empty, log an error. And skip this date.
        if len(route) < 1:
            dpr.LOGGERS['main'].error(
                f"{vehicle_id} has no edges in its route on {date}!")
            route = []
            for stop_node in stop_nodes:
                route.append(stop_node.get('lane').split('_')[0])
        else:
            route.append(last_found_edge.getID())

        route_str = " ".join(str(edge) for edge in route)
        route_node = et.Element('route', {'edges': route_str})
        vehicle_node.append(route_node)

        for stop_node in stop_nodes:
            vehicle_node.append(stop_node)

        root.append(vehicle_node)

    return tree

# TODO Create file with statistics regarding gps coordinates which failed to
# find a nearby edge.


def _do_route_building(input_file: Path, output_dir: Path, xml_template: Path,
                       skip_existing: bool, net: sumolib.net.Net,
                       stop_labels: t.List[bool]):
    """Route building multiprocessing function"""
    try:
        input_data = pd.read_csv(input_file)
    except pd.errors.EmptyDataError:
        dpr.LOGGERS['main'].error(
            f"No data in file: {input_file.name}. Skipping...")
        return

    # Create a subpath in the output directory with the name of the current
    # taxi.
    #   TODO: Get the name using some other method. Rather than using the
    #   parent directory, use some way to store metadata.
    output_subdir = output_dir.joinpath(input_file.parent.name)
    output_subdir.mkdir(parents=True, exist_ok=True)
    export_file = output_subdir.joinpath(f"{input_file.stem}.rou.xml")
    if skip_existing and export_file.exists():
        dpr.LOGGERS.info(f"{export_file.stem}.rou.xml exists. Skipping...")
        return

    tree = generate_route_xml(net, input_data, xml_template, input_file.stem,
                              stop_labels)
    # export the xml tree as a file, using minidom to create a "pretty" xml
    # from our element tree
    #   TODO Remove the random spaces. (in the generated xml file)
    xmlstr = minidom.parseString(
        et.tostring(tree.getroot())).toprettyxml(indent="    ")
    with open(export_file, 'w') as f:
        f.write(xmlstr)

    # Create additionals file with vType definition, if it doesn't exist
    # already.
    tree = et.parse(xml_template)
    root = tree.getroot()
    vType_node = root.find('vType')
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


# XXX Deprecated: Remove this function.
def _do_route_building_old(trip_file: Path, output_dir: Path, skip_existing: bool,
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

    # Get the vType section of xml.
    vType_node = trip_tree_root.find('vType')

    # Create a list of pairs of stops [(s1, s2), (s2, s3), ...]
    trip_node = trip_tree_root.find('trip')
    first_stop = trip_node.attrib['from']  # s1
    last_stop = trip_node.attrib['to']  # sN

    # Get stops inbetween the first and the last stop of the trip.
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
            dpr.LOGGERS.error(
                f"{ev_name}: {date}: Failed to find a route "
                f"between edges {stop_pair[0]} and {stop_pair[1]}")
            # Assign the last edge of the stop_pair to `route edges`, *just in-
            # case* this is the last iteration of the loop. If `route_edges` is
            # not assigned, `route_edges[-1].getID()` (see a few lines below)
            # will throw an error.
            route_edges = [net.getEdge(stop_pair[-1])]
    # if chained route is empty, return. Don't continue saving the xml file.
    if len(route) < 1:
        dpr.LOGGERS['main'].error(f"{ev_name} has no edges in its route on {date}!")
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
    #         logger.error("DuaIterate failed at: \n\t" +
    #                       str(output_subdir.absolute()))
    #     # Restore current_dir
    #     os.chdir(cwd)


def build_routes(scenario_dir: Path, **kwargs):
    """
    Run the route building procedure. The "main" function of this script
    """

    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])

    cluster_dir = scenario_dir.joinpath('Spatial_Clusters')
    input_list = sorted(
        [*cluster_dir.joinpath("Filtered_Traces").glob("*/*.csv")])
    output_dir = scenario_dir.joinpath("Mobility_Simulation", "Routes")
    xml_template = scenario_dir.joinpath('_Inputs', 'Configs',
                                         'ev_template.xml')

    # Load stop_labels
    stop_labels_dir = scenario_dir.joinpath('Temporal_Clusters', 'Stop_Labels')
    stop_labels_files = sorted([*stop_labels_dir.glob('*.csv')])
    stop_labels_all = []
    for stop_labels_file in stop_labels_files:
        stop_labels_ev = pd.read_csv(stop_labels_file)
        # Convert the timecolumn to datetimes or timedeltas.
        if input_data_fmt == dpr.DATA_FMTS['GPS']:
            stop_labels_ev['Time'] = pd.to_datetime(stop_labels_ev['Time'])
        elif input_data_fmt == dpr.DATA_FMTS['GTFS']:
            stop_labels_ev['Time'] = pd.to_timedelta(stop_labels_ev['Time'])
        else: raise ValueError(dpr.DATA_FMT_ERROR_MSG)

        # Make the time column the index.
        stop_labels_ev = stop_labels_ev.set_index('Time')
        # Take the time index.
        time_index = stop_labels_ev.index.get_level_values('Time')
        # Set the dates in the time index as the new index.
        if input_data_fmt == dpr.DATA_FMTS['GPS']:
            stop_labels_ev = stop_labels_ev.set_index(time_index.date)
        elif input_data_fmt == dpr.DATA_FMTS['GTFS']:
            stop_labels_ev = stop_labels_ev.set_index(time_index.days)
        else: raise ValueError(dpr.DATA_FMT_ERROR_MSG)
        # Rename the index to 'Date'
        stop_labels_ev.index.names = ['Date']

        # Seperate the dataframe by date.
        # For each date in the index:
        if input_data_fmt == dpr.DATA_FMTS['GPS']:
            for date in sorted(set(time_index.date)):
                # Append the stop_labels on that date to the stop_labels_all list.
                stop_labels_all.append(stop_labels_ev.loc[[date]])
        elif input_data_fmt == dpr.DATA_FMTS['GTFS']:
            for date in sorted(set(time_index.days)):
                # Append the stop_labels on that date to the stop_labels_all list.
                stop_labels_all.append(stop_labels_ev.loc[[date]])
        else: raise ValueError(dpr.DATA_FMT_ERROR_MSG)

    # Ask user if s/he wants to skip existing rou.xml files
    _ = dpr.auto_input("Would you like to skip existing rou.xml files? y/[n]",
                       'n', **kwargs)
    skip_existing = True if _.lower() == 'y' else False

    dpr.LOGGERS['main'].info("Loading sumo network...")
    try:
        network_file = [
            *scenario_dir.joinpath('_Inputs', 'Map').glob('*.net.xml')
        ][0]
    except IndexError:
        dpr.LOGGERS['main'].error(
            "You have not created a SUMO network in: \n\t" +
            str(scenario_dir.joinpath('_Inputs', 'Map')))
        sys.exit(1)

    net = sumolib.net.readNet(network_file)
    print("Done loading network.\n")

    print("Generating route files...")

    # OLD MULTITHREADED WAY:
    # ----------------------
    # """ Doesn't work because it has too much recursive depth. """
    # args_rou = zip(
    #     input_list,
    #     repeat(output_dir, len(input_list)),
    #     repeat(xml_template, len(input_list)),
    #     repeat(skip_existing, len(input_list)),
    #     repeat(net, len(input_list)))
    # with Pool(cpu_count() - 3) as p:
    #     p.starmap(_do_route_building, args_rou)

    # Do route generation
    for input_file, stop_labels_ev in tqdm(zip(input_list, stop_labels_all)):
        _do_route_building(input_file, output_dir, xml_template,
                           skip_existing, net, stop_labels_ev)
    print("Done generating route files.")

    # # Ask user if s/he wants to skip existing rou.xml files
    # _ = dpr.auto_input("Would you like to skip existing rou.xml files? y/[n]",
    #                    'n', **kwargs)
    # skip_existing = True if _.lower() == 'y' else False

    # print("Generating route files...")
    # # Do route generation
    # rou_files = [*output_dir.glob('*/*.rou.xml')]
    # output_dir = scenario_dir.joinpath("Mobility_Simulation", "Routes")

    # # OLD MULTITHREADED WAY:
    # # ----------------------
    # # """ Doesn't work because it has too much recursive depth. """
    # # args_route = zip(
    # #     rou_files,
    # #     repeat(output_dir, len(input_list)),
    # #     repeat(skip_existing, len(input_list)),
    # #     repeat(net, len(input_list)))
    # # with Pool(cpu_count() - 3) as p:
    # #     p.starmap(_do_route_building, args_route)

    # # IF DEBUGGING
    # # ------------
    # for rou_file in tqdm(rou_files):
    #     _do_route_building_old(rou_file, output_dir, skip_existing, net)
    # print("Done generating route files.")

    # Ask user if s/he wants to skip combinding rou.xml files
    #   TODO TODO Allow for combining according to an arbitrary number of
    # files. For example, I have 12 cpu threads, so I would like to group them
    # into 11 files so that I can simulate each of them and split each of their
    # results simultaneously. (Although, a lot of ram would be required to
    # simulate them simultaneously...)
    #   TODO Ask this in main.py and only if `configuring` = True
    COMBINING_METHODS = {'per_ev': 1, 'all_evs': 2, 'skip': 3}

    print("How would you like to combine the routes for this simulation " +
          "scenario?\n")
    if input_data_fmt == dpr.DATA_FMTS['GPS']:
        print(
            "[1]. Create one SUMO simulation which combines the routes for each EV.\n"
            " 2 . Create one SUMO for which combines *all* the routes across *all* EVs.\n"
            " 3 . Skip combining the routes entirely.\n")
    elif input_data_fmt == dpr.DATA_FMTS['GTFS']:
        print(
            " 1 . Create one SUMO simulation which combines the routes for each EV.\n"
            "[2]. Create one SUMO for which combines *all* the routes across *all* EVs.\n"
            " 3 . Skip combining the routes entirely.\n")
    else:
        raise ValueError(dpr.DATA_FMT_ERROR_MSG)

    _ = dpr.auto_input("Enter a number:  ", '', **kwargs)

    if _ == '1':
        combining_method = COMBINING_METHODS['per_ev']
    elif _ == '2':
        combining_method = COMBINING_METHODS['all_evs']
    elif _ == '3':
        combining_method = COMBINING_METHODS['skip']
    else:
        if input_data_fmt == dpr.DATA_FMTS['GPS']:
            combining_method = COMBINING_METHODS['per_ev']
        elif input_data_fmt == dpr.DATA_FMTS['GTFS']:
            combining_method = COMBINING_METHODS['all_evs']
        else:
            raise ValueError(dpr.DATA_FMT_ERROR_MSG)

    # TODO Combine the routes on a per-taxi basis. Therefore, if there are 10
    # taxis, and I have 4 cpu threads, I can run 4 sumo simulations in parallel.
    if combining_method == COMBINING_METHODS['per_ev']:
        print("Combining routes into one file per EV.")
        # Combine all routes into one file per EV!
        ev_dirs = []
        for child in [*scenario_dir.joinpath("Mobility_Simulation", "Routes").glob('*/')]:
            if child.is_dir():
                ev_dirs.append(child)
        output_dir = scenario_dir.joinpath("Mobility_Simulation", "Combined_Routes")
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

    elif combining_method == COMBINING_METHODS['all_evs']:
        print("Combining routes into one for all EVs.")
        # Combine all routes into one file for all EVs
        route_files = [*scenario_dir.joinpath("Mobility_Simulation",
                                              "Routes").glob('*/*.rou.xml')]
        output_dir = scenario_dir.joinpath("Mobility_Simulation", "Combined_Routes")
        output_dir.mkdir(parents=True, exist_ok=True)

        route_files.sort()
        combined_rou_xml_file = output_dir.joinpath(
            'monolithic.rou.xml')
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
        print("Done combining routes for all the EVs.")

    elif combining_method == COMBINING_METHODS['skip']:
        print("Skpping route combining.")

    else:
        raise ValueError("Invalid combining method...")
