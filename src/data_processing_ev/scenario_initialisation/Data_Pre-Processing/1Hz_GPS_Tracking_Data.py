#!/usr/bin/env python3

"""
For use on the following dataset:
https://data.mendeley.com/datasets/xt69cnwh56/1
"""


import os
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from operator import itemgetter
import numpy as np
import csv

Lat_max = -180.000000
Lon_max = -180.000000
Lon_min = 180.000000
Lat_min = 180.0000000


def _generate_traces(traces_dir: Path):
    # For each file, identify the vehicle_id corresponding to it and the
    # start-date of the file's data.
    original_files = sorted([*traces_dir.joinpath('Original').glob('*.csv')])

    # For each vehicle_id:
    for original_file in tqdm(original_files):
        output_file = traces_dir.joinpath('Processed_1Hz', original_file.name)
        # Write the header row.
        header_row = ("GPSID,Time,Latitude,Longitude,Altitude,Heading,Satellites,HDOP,AgeOfReading,DistanceSinceReading,Velocity")
        output_file.parent.mkdir(parents=True, exist_ok=True)
         # For each file in files_of_vehicle:
        with open(original_file, 'r') as f_in:
            # Read and discard the header row.
            reader = csv.reader(f_in)
            header = next(reader)
            # For each remaining row in the file:
            with open(output_file, 'w') as f_out:
                f_out.write(header_row + '\n')


                for row in reader:
                    date = row[0]
                    time = row[1]
                    latitude = row[2]
                    longitude = row[3]
                    altitude = row [4]
                    speed = row[5]
                    heading = row[6]
                    satellites = row[8]


                    global Lat_max
                    if float(latitude) > float(Lat_max):
                        Lat_max = float(latitude) + 0.000200

                    global Lon_max
                    if float(longitude) > float(Lon_max):
                        Lon_max = float(longitude) + 0.000200

                    global Lat_min
                    if float(latitude) < float(Lat_min):
                        Lat_min = float(latitude) - 0.000200

                    global Lon_min
                    if float(longitude) < float(Lon_min):
                        Lon_min = float(longitude) - 0.000200

                    length = len(date)
                    year = date[length-4]+date[length-3]+date[length-2]+date[length-1]

                    if date[length-7]=="/": 						#Month is a single figure
                        month = "0"+date[length-6]
                        #we know date[length-7] = /
                        if length==8:			 					#Day is a single figure
                            day = "0"+date[length-8]
                        elif length==9:		  						#Day is a double figure
                            day = date[length-9]+date[length-8]

                    elif date[length-8]=="/":						#Month is a double figure
                        month = date[length-6]+date[length-7]
                        #we know date[lenght-8] = /
                        if length==9:		  						#Day is a single figure
                            day = "0"+date[length-9]
                        elif length==10:
                            day = date[length-10]+date[length-9]	#Day is a double figure

                    if time[1]==":":         #Hour is a single digit
                        time= "0"+time

                    time = time[0]+time[1]+time[2]+time[3]+time[4]+time[5]+time[6]+time[7]
                    if int(month) > 12:
                        month = month[1] + month[0]

                    line = "0,{}-{}-{} {},{},{},{},{},{},0,0,0,{}".format(year,month,day,time,latitude,longitude,altitude,heading,satellites,speed) + "\n"
                    f_out.write(line)


def _generate_minute_data(traces_dir: Path):
    original_files = sorted([*traces_dir.joinpath('Processed_1Hz').glob('*.csv')])

    for original_file in tqdm(original_files):
        output_file = traces_dir.joinpath('Processed', original_file.name)
        header_row = ("GPSID,Time,Latitude,Longitude,Altitude,Heading,Satellites,HDOP,AgeOfReading,DistanceSinceReading,Velocity")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(original_file, 'r') as f_in:
            # Read and discard the header row.
            reader = csv.reader(f_in)
            header = next(reader)



            with open(output_file, 'w') as f_out:
                f_out.write(header_row + '\n')
                previous_minute=None
                for row in reader:
                    gpsid = row[0]
                    time = row[1]
                    lat = row[2]
                    lon = row[3]
                    alt = row[4]
                    heading = row[5]
                    satellites = row[6]
                    hdop = row[7]
                    AOR = row[8]
                    DSR = row[9]
                    speed = row[10]


                    current_minute = time[15]

                    if (current_minute != previous_minute):
                        new_minute = True
                        previous_minute = current_minute

                    else:
                        new_minute = False

                    if new_minute == True:
                        line = "{},{},{},{},{},{},{},{},{},{},{}".format(gpsid,time,lat,lon,alt,heading,satellites,hdop,AOR,DSR,speed) + "\n"
                        f_out.write(line)

                line = "{},{},{},{},{},{},{},{},{},{},{}".format(gpsid,time,lat,lon,alt,heading,satellites,hdop,AOR,DSR,speed) + "\n"
                f_out.write(line)






def main(scenario_dir: Path):
    # Create a list of csv files found in the traces directory.
    traces_dir = scenario_dir.joinpath('_Inputs', 'Traces')

    traces_dir.joinpath('Processed_1Hz').mkdir(parents=True, exist_ok=True)

    if not any(traces_dir.joinpath('Processed').glob('*.csv')):
        _generate_traces(traces_dir)
    else:
        # Else (if there are aleardy processed csv files):
        _ = input("(Re)generate processed traces? [y/N] ")
        if _.lower() == 'y':
            _generate_traces(traces_dir)

    _generate_minute_data(traces_dir)

    traces_dir = scenario_dir.joinpath('_Inputs', 'Map')

    output_file = traces_dir.joinpath('Boundary', "boundary.csv")


    with open(output_file, 'w') as outfile:
        line = "Longitude,Latitude\n"
        outfile.write(line)
        line = "{},{}\n".format(Lon_min,Lat_min)
        outfile.write(line)
        line = "{},{}\n".format(Lon_max,Lat_min)
        outfile.write(line)
        line = "{},{}\n".format(Lon_max,Lat_max)
        outfile.write(line)
        line = "{},{}\n".format(Lon_min,Lat_max)
        outfile.write(line)


if __name__ == '__main__':
    scenario_dir = Path(os.path.abspath(__file__)).parents[2]  # XXX This isn't working when pdb is loaded...
    main(scenario_dir)


