from __future__ import print_function
import csv
import itertools
import sys
import datetime
import os
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
from pyparsing import line_end
from tqdm import tqdm
from operator import itemgetter
import numpy as np
import csv
import geopy.distance
from geopy.distance import geodesic as GD
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import save
from numpy import savetxt
from statistics import mean
from math import radians, cos, sin, asin, sqrt

g = 9.81
p = 1.184
offtake = 100
mv = 3900
cd = 0.36
crr = 0.02
area = 4
power_eff = 0.9
brake_eff = 0.65

efficiencies = []
distances = []
time_diff = []
max_engine_power_array = []


def _generate_traces(traces_dir: Path):

    original_files = sorted([*traces_dir.joinpath('Input_Trips').glob('*.csv')])

    with open('Results.csv', 'w') as f_results:
        header = ("Trip,Max Engine Power (kW), Energy (kWh), Distance (km), Energy Per Distance (kWh/km)")
        f_results.write(header + '\n')

    for original_file in tqdm(original_files):

        output_file = traces_dir.joinpath('Processed_Trips', original_file.name)
        
        header_row = ("Time,Latitude,Longitude,Altitude,Velocity,Time Diff (s),Displacement (m),Engine Power (kW),Propulsion Energy (Wh),Braking Energy (Wh),Offtake Energy (Wh)")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        total_energy = 0
        total_displacement = 0
        max_engine_power = 0
        
        with open(original_file, 'r') as f_in:
            
            reader = csv.reader(f_in)
            header = next(reader)
            first_row = next(reader)

            
            
            date = first_row[0]
            previous_lat = float(first_row[1])
            previous_lon = float(first_row[2])
            previous_alt = float(first_row[3])
            previous_speed = float(first_row[4])
            if previous_speed < 0.5:
                previous_speed = 0
            else:
                previous_speed = previous_speed/3.6

            month = date[5]+date[6]
            if month[0]=='0':
                month = month[1]
            

            day = date[8]+date[9]
            if day[0]=='0':
                day = day[1]
            day = float(day)

            hour = date[11]+date[12]
            if hour[0]=='0':
                hour = hour[1]
            hour = float(hour)

            minute = date[14]+date[15]
            if minute[0]=='0':
                minute = minute[1]
            minute = float(minute)

            second = date[17]+date[18]
            if second[0]=='0':
                second = second[1]
            second = float(second)


            if month=='1':
                monthly_hours = 0
            elif month=='2':
                monthly_hours = 744
            elif month=='3':
                monthly_hours = 1416
            elif month=='4':
                monthly_hours = 2160
            elif month=='5':
                monthly_hours = 2880
            elif month=='6':
                monthly_hours = 3624
            elif month=='7':
                monthly_hours = 4344
            elif month=='8':
                monthly_hours = 5088
            elif month=='9':
                monthly_hours = 5832
            elif month=='10':
                monthly_hours = 6552
            elif month=='11':
                monthly_hours = 7296
            elif month=='12':
                monthly_hours = 8016
            
            monthly_seconds = monthly_hours*60*60
            daily_seconds = day*24*60*60

            if date[20]=='A' and hour==12:
                hourly_hours = 0
            elif date[20]=='A':
                hourly_hours = hour
            elif date[20]=='P' and hour==12:
                hourly_hours = 12
            elif date[20]=='P':
                hourly_hours = 12 + hour

            hourly_seconds = hourly_hours*60*60
            
            minutely_seconds = minute*60

            previous_seconds = monthly_seconds + daily_seconds + hourly_seconds + minutely_seconds + second


            with open(output_file, 'w') as f_out:
                f_out.write(header_row + '\n')
                

                for row in reader:
                    date = row[0]
                    lat = float(row[1])
                    lon = float(row[2])
                    alt = float(row[3])
                    speed = float(row[4])
                    if speed < 0.5:
                        speed = 0
                    else:
                        speed = speed/3.6

                    month = date[5]+date[6]
                    if month[0]=='0':
                        month = month[1]
                    

                    day = date[8]+date[9]
                    if day[0]=='0':
                        day = day[1]
                    day = float(day)

                    hour = date[11]+date[12]
                    if hour[0]=='0':
                        hour = hour[1]
                    hour = float(hour)

                    minute = date[14]+date[15]
                    if minute[0]=='0':
                        minute = minute[1]
                    minute = float(minute)

                    second = date[17]+date[18]
                    if second[0]=='0':
                        second = second[1]
                    second = float(second)


                    if month=='1':
                        monthly_hours = 0
                    elif month=='2':
                        monthly_hours = 744
                    elif month=='3':
                        monthly_hours = 1416
                    elif month=='4':
                        monthly_hours = 2160
                    elif month=='5':
                        monthly_hours = 2880
                    elif month=='6':
                        monthly_hours = 3624
                    elif month=='7':
                        monthly_hours = 4344
                    elif month=='8':
                        monthly_hours = 5088
                    elif month=='9':
                        monthly_hours = 5832
                    elif month=='10':
                        monthly_hours = 6552
                    elif month=='11':
                        monthly_hours = 7296
                    elif month=='12':
                        monthly_hours = 8016
                    
                    monthly_seconds = monthly_hours*60*60
                    daily_seconds = day*24*60*60

                    if date[20]=='A' and hour==12:
                        hourly_hours = 0
                    elif date[20]=='A':
                        hourly_hours = hour
                    elif date[20]=='P' and hour==12:
                        hourly_hours = 12
                    elif date[20]=='P':
                        hourly_hours = 12 + hour

                    hourly_seconds = hourly_hours*60*60
                    minutely_seconds = minute*60
                    seconds = monthly_seconds + daily_seconds + hourly_seconds + minutely_seconds + second

                    delta_t = seconds-previous_seconds #seconds
                    delta_v = speed - previous_speed
                    elev_change = alt-previous_alt
                    if abs(elev_change) < 0.2:
                        elev_change = 0

                    previous_location = [previous_lat,previous_lon]
                    current_location = [lat,lon]
                    dist_lateral = geopy.distance.geodesic(previous_location,current_location).m
                    dist_3D = np.sqrt(dist_lateral**2 + elev_change**2)

                    if dist_3D != 0 and elev_change != 0:
                        slope = np.arcsin(elev_change/dist_3D)
                    else:
                        slope = 0
                    
                    

                    Frr = 0
                    Fad = 0
                    Fsd = 0
                    force = 0

                    if previous_speed != 0 :
                        if previous_speed > 0.3:
                            Frr = -mv*g*crr*np.cos(slope)
                        Fad = -0.5*p*cd*area*previous_speed**2
                        Fsd = -mv*g*np.sin(slope)

                    

                    force = Frr + Fad + Fsd

                    exp_speed_delta = force*delta_t/mv
                    unexp_speed_delta = delta_v - exp_speed_delta

                    try:
                        prop_brake_force = mv*unexp_speed_delta/delta_t
                        kinetic_power = prop_brake_force*previous_speed
                        propulsion_work = kinetic_power*delta_t
                    except ZeroDivisionError:
                        prop_brake_force = 0
                        kinetic_power = 0
                        propulsion_work = 0

                    Er = propulsion_work
                    ErP = 0
                    ErB = 0

                    if Er > 0:
                        ErP = Er/power_eff #Joules
                        engine_power = kinetic_power/power_eff/1000
                    elif Er < 0:
                        ErB = Er*brake_eff #Joules

                    ErOfftake = delta_t*offtake

                    Energy_Consumption = (ErP + ErOfftake + ErB)/3.6e6 #kWh

                    

                    if engine_power > max_engine_power:
                        max_engine_power = engine_power

                    total_energy = total_energy + Energy_Consumption #kWh
                    total_displacement = total_displacement + dist_3D #m

                    line = "{},{},{},{},{},{},{},{},{},{},{}".format(date,lat,lon,alt,speed,delta_t,dist_3D,engine_power,ErP,ErB,ErOfftake) + "\n"
                    f_out.write(line)

                    previous_seconds = seconds
                    previous_alt = alt
                    previous_lat = lat
                    previous_lon = lon
                    previous_speed = speed

        energy_per_distance = (total_energy)/(total_displacement/1000) #kWh/km
        efficiencies.append(energy_per_distance)

        total_energy_kWh = total_energy/1000
        total_displacement_km = total_displacement/1000

        distances.append(total_displacement_km)
        max_engine_power_array.append(max_engine_power)

        with open('Results.csv', 'a') as f_results:
            line = "{},{},{},{},{}".format(original_file.name,max_engine_power,total_energy_kWh,total_displacement_km,energy_per_distance)
            f_results.write(line + '\n')


def _generate_traces_Trailer(traces_dir: Path):
    mv = 5710
    crr = 0.03
    cd = 0.42

    original_files = sorted([*traces_dir.joinpath('Input_Trips').glob('*.csv')])

    with open('Results_Trailer.csv', 'w') as f_results:
        header = ("Trip,Max Engine Power (kW), Energy (kWh), Distance (km), Energy Per Distance (kWh/km)")
        f_results.write(header + '\n')

    for original_file in tqdm(original_files):

        output_file = traces_dir.joinpath('Processed_Trips_Trailer', original_file.name)
        
        header_row = ("Time,Latitude,Longitude,Altitude,Velocity,Time Diff (s),Displacement (m),Engine Power (kW),Propulsion Energy (Wh),Braking Energy (Wh),Offtake Energy (Wh)")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        total_energy = 0
        total_displacement = 0
        max_engine_power = 0
        
        with open(original_file, 'r') as f_in:
            
            reader = csv.reader(f_in)
            header = next(reader)
            first_row = next(reader)

            
            
            date = first_row[0]
            previous_lat = float(first_row[1])
            previous_lon = float(first_row[2])
            previous_alt = float(first_row[3])
            previous_speed = float(first_row[4])
            if previous_speed < 0.5:
                previous_speed = 0
            else:
                previous_speed = previous_speed/3.6

            month = date[5]+date[6]
            if month[0]=='0':
                month = month[1]
            

            day = date[8]+date[9]
            if day[0]=='0':
                day = day[1]
            day = float(day)

            hour = date[11]+date[12]
            if hour[0]=='0':
                hour = hour[1]
            hour = float(hour)

            minute = date[14]+date[15]
            if minute[0]=='0':
                minute = minute[1]
            minute = float(minute)

            second = date[17]+date[18]
            if second[0]=='0':
                second = second[1]
            second = float(second)


            if month=='1':
                monthly_hours = 0
            elif month=='2':
                monthly_hours = 744
            elif month=='3':
                monthly_hours = 1416
            elif month=='4':
                monthly_hours = 2160
            elif month=='5':
                monthly_hours = 2880
            elif month=='6':
                monthly_hours = 3624
            elif month=='7':
                monthly_hours = 4344
            elif month=='8':
                monthly_hours = 5088
            elif month=='9':
                monthly_hours = 5832
            elif month=='10':
                monthly_hours = 6552
            elif month=='11':
                monthly_hours = 7296
            elif month=='12':
                monthly_hours = 8016
            
            monthly_seconds = monthly_hours*60*60
            daily_seconds = day*24*60*60

            if date[20]=='A' and hour==12:
                hourly_hours = 0
            elif date[20]=='A':
                hourly_hours = hour
            elif date[20]=='P' and hour==12:
                hourly_hours = 12
            elif date[20]=='P':
                hourly_hours = 12 + hour

            hourly_seconds = hourly_hours*60*60
            
            minutely_seconds = minute*60

            previous_seconds = monthly_seconds + daily_seconds + hourly_seconds + minutely_seconds + second


            with open(output_file, 'w') as f_out:
                f_out.write(header_row + '\n')
                

                for row in reader:
                    date = row[0]
                    lat = float(row[1])
                    lon = float(row[2])
                    alt = float(row[3])
                    speed = float(row[4])
                    if speed < 0.5:
                        speed = 0
                    else:
                        speed = speed/3.6

                    month = date[5]+date[6]
                    if month[0]=='0':
                        month = month[1]
                    

                    day = date[8]+date[9]
                    if day[0]=='0':
                        day = day[1]
                    day = float(day)

                    hour = date[11]+date[12]
                    if hour[0]=='0':
                        hour = hour[1]
                    hour = float(hour)

                    minute = date[14]+date[15]
                    if minute[0]=='0':
                        minute = minute[1]
                    minute = float(minute)

                    second = date[17]+date[18]
                    if second[0]=='0':
                        second = second[1]
                    second = float(second)


                    if month=='1':
                        monthly_hours = 0
                    elif month=='2':
                        monthly_hours = 744
                    elif month=='3':
                        monthly_hours = 1416
                    elif month=='4':
                        monthly_hours = 2160
                    elif month=='5':
                        monthly_hours = 2880
                    elif month=='6':
                        monthly_hours = 3624
                    elif month=='7':
                        monthly_hours = 4344
                    elif month=='8':
                        monthly_hours = 5088
                    elif month=='9':
                        monthly_hours = 5832
                    elif month=='10':
                        monthly_hours = 6552
                    elif month=='11':
                        monthly_hours = 7296
                    elif month=='12':
                        monthly_hours = 8016
                    
                    monthly_seconds = monthly_hours*60*60
                    daily_seconds = day*24*60*60

                    if date[20]=='A' and hour==12:
                        hourly_hours = 0
                    elif date[20]=='A':
                        hourly_hours = hour
                    elif date[20]=='P' and hour==12:
                        hourly_hours = 12
                    elif date[20]=='P':
                        hourly_hours = 12 + hour

                    hourly_seconds = hourly_hours*60*60
                    minutely_seconds = minute*60
                    seconds = monthly_seconds + daily_seconds + hourly_seconds + minutely_seconds + second

                    delta_t = seconds-previous_seconds #seconds
                    delta_v = speed - previous_speed
                    elev_change = alt-previous_alt
                    if abs(elev_change) < 0.2:
                        elev_change = 0

                    previous_location = [previous_lat,previous_lon]
                    current_location = [lat,lon]
                    dist_lateral = geopy.distance.geodesic(previous_location,current_location).m
                    dist_3D = np.sqrt(dist_lateral**2 + elev_change**2)

                    if dist_3D != 0 and elev_change != 0:
                        slope = np.arcsin(elev_change/dist_3D)
                    else:
                        slope = 0
                    
                    

                    Frr = 0
                    Fad = 0
                    Fsd = 0
                    force = 0

                    if previous_speed != 0 :
                        if previous_speed > 0.3:
                            Frr = -mv*g*crr*np.cos(slope)
                        Fad = -0.5*p*cd*area*previous_speed**2
                        Fsd = -mv*g*np.sin(slope)

                    

                    force = Frr + Fad + Fsd

                    exp_speed_delta = force*delta_t/mv
                    unexp_speed_delta = delta_v - exp_speed_delta

                    try:
                        prop_brake_force = mv*unexp_speed_delta/delta_t
                        kinetic_power = prop_brake_force*previous_speed
                        propulsion_work = kinetic_power*delta_t
                    except ZeroDivisionError:
                        prop_brake_force = 0
                        kinetic_power = 0
                        propulsion_work = 0

                    Er = propulsion_work
                    ErP = 0
                    ErB = 0

                    if Er > 0:
                        ErP = Er/power_eff #Joules
                        engine_power = kinetic_power/power_eff/1000
                    elif Er < 0:
                        ErB = Er*brake_eff #Joules

                    ErOfftake = delta_t*offtake

                    Energy_Consumption = (ErP + ErOfftake + ErB)/3.6e6 #kWh

                    

                    if engine_power > max_engine_power:
                        max_engine_power = engine_power

                    total_energy = total_energy + Energy_Consumption #kWh
                    total_displacement = total_displacement + dist_3D #m

                    line = "{},{},{},{},{},{},{},{},{},{},{}".format(date,lat,lon,alt,speed,delta_t,dist_3D,engine_power,ErP,ErB,ErOfftake) + "\n"
                    f_out.write(line)

                    previous_seconds = seconds
                    previous_alt = alt
                    previous_lat = lat
                    previous_lon = lon
                    previous_speed = speed

        energy_per_distance = (total_energy)/(total_displacement/1000) #kWh/km
        efficiencies.append(energy_per_distance)

        total_energy_kWh = total_energy/1000
        total_displacement_km = total_displacement/1000

        distances.append(total_displacement_km)
        max_engine_power_array.append(max_engine_power)

        with open('Results.csv', 'a') as f_results:
            line = "{},{},{},{},{}".format(original_file.name,max_engine_power,total_energy_kWh,total_displacement_km,energy_per_distance)
            f_results.write(line + '\n')


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def _boxplot():

    _average_energy_perdistance = mean(efficiencies)
    data = [efficiencies]
    #save('data.npy', data)


    ticks = ['']
    plt.figure()

    bpl = plt.boxplot(data, positions=np.array(range(len(data)))*2.0, sym='', widths=0.6, showfliers=True, showmeans=True)
    
    set_box_color(bpl, '#D7191C')


    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-0.5, 0.5)
    plt.ylim(0, 0.6)
    plt.ylabel('Energy Expenditure [kWh/km]')
    plt.xlabel('All 48 Long Distance Trips')
    plt.tight_layout()
    plt.show()

def _simplified_boxplot():

    data = [efficiencies]
    
    fig = plt.figure(figsize =(10, 7))
    
    # Creating plot
    plt.boxplot(data,showfliers=True, showmeans=True)
    
    # show plot
    plt.ylim(0.6, 1)
    plt.ylabel('Energy Expenditure [kWh/km]')
    plt.xlabel('All 48 Long Distance Trips')
    plt.show()

def _simplified_boxplot_time():

    data = [time_diff]
    
    fig = plt.figure(figsize =(10, 7))
    
    # Creating plot
    plt.boxplot(data,showfliers=True, showmeans=True)
    
    # show plot
    plt.ylim(-10, 1000)
    plt.ylabel('Time Difference Between Samles [s]')
    plt.xlabel('All 48 Long Distance Trips')
    plt.show()

def _simplified_boxplot_engine_power():

    data = [max_engine_power_array]
    
    fig = plt.figure(figsize =(10, 7))
    
    # Creating plot
    plt.boxplot(data,showfliers=True, showmeans=True)
    
    # show plot
    plt.ylim(0, 300)
    plt.ylabel('Max Engine Power In Dataset [kW]')
    plt.xlabel('All 48 Long Distance Trips')
    plt.show()

def main(scenario_dir: Path):
    # Create a list of csv files found in the traces directory.
    traces_dir = scenario_dir.joinpath('Swap_Article', 'Kinetic_Model')
    #_generate_traces(traces_dir)
    _generate_traces_Trailer(traces_dir)
    #_boxplot()
    #_simplified_boxplot()
    #_simplified_boxplot_time()
    #_simplified_boxplot_engine_power()
    print("Mean kWh/km: ")
    print(mean(efficiencies))
    #print(mean(max_engine_power_array))
    #print("Mean km/trip: ")
    #print(mean(distances))
    #print("Time Diff: ")
    #print(mean(time_diff))



if __name__ == '__main__':
    scenario_dir = Path(os.path.abspath(__file__)).parents[2]  # XXX This isn't working when pdb is loaded...
    main(scenario_dir)


