"""This function intakes a pandas dataframe of trip data ('journey'), and adds
columns with the geodesic distance and slope angles between consecutive
observations"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import geopy.distance


def read_file(filename, path):
    """ This function takes the filename of a GPS trip stored in a .csv file,
    and path to the file.

    It reads the file, cleans it, adds columns that are necessary for the
    kinetic model, and returns the file in a pandas dataframe. """

    journey = pd.read_csv(path + "/" + filename)

    # Data cleaning

    if 'Aatitude' in journey.columns:
        journey['Altitude'] = journey['Aatitude']
        journey.drop(columns = ['Aatitude'])

    if 'GPS Speed' in journey.columns:
        journey['Speed'] = journey['GPS Speed']
        journey.drop(columns = ['GPS Speed'])

    # Join date and time columns into one column
    journey['DateTime'] = pd.to_datetime(journey['Date'] + journey['Time'], format = '%m/%d/%Y%H:%M:%S')

    # Check if the time logger got stuck for a second, and correct if so
    for i in range(len(journey) - 1):
         if journey['DateTime'][i] == journey['DateTime'][i + 1]:
            journey['DateTime'][i + 1] += timedelta(seconds = 1)



    #######################################################################

    ############ Set up dataframe for energy consumption estimations ############


    # Convert speed in km/h to m/s
    journey['Velocity'] = np.where(journey['Speed'] >= 0.5, journey['Speed']/3.6, 0) #


   # for i in range(len(journey)): ### Observations less than 0.5 km/h are set to 0, due to GPS noise
   #     if journey['Speed'][i] < 0.5:
      #      journey['Velocity'][i] = 0

    #Calculate elevation change
    journey['ElevChange'] = np.where(abs(journey['Altitude'].shift(-1) - journey['Altitude']) >= 0.2, journey['Altitude'].shift(-1) - journey['Altitude'], 0)
   # for i in range(len(journey)): # If measured elevation change < 0.2, then set to 0 due to GPS noise.
    #    if abs(journey['ElevChange'][i]) < 0.2:
     #       journey['ElevChange'][i] = 0

    # Calculate time between samples. Useful for kinetic model.
    # This is needed because some samples are 1/2Hz.
    journey['DeltaT'] = journey['DateTime'].shift(-1) - journey['DateTime']
    journey.DeltaT = journey['DeltaT']/ np.timedelta64(1, 's') # Convert from timedelta to float

    # Calculate change in velocity between each timestep
    journey['DeltaV'] = journey['Velocity'].shift(-1) - journey['Velocity']

    # Calculate acceleration
    journey['Acceleration'] = np.where(journey['DeltaT'] > 0, journey['DeltaV']/journey['DeltaT'], 0)

    # Joins lat/lon Coords into one column, useful for getDist function
    journey['Coordinates'] = list(zip(journey.Latitude, journey.Longitude))

    return journey

def getDistSlope(journey):
    """
    Inputs: dataframe with vehicle journey data

    Calculates distance between each successive pair of lat/lon coordinates
    (accounts for elevation difference)

    Calculates slope angle faced by vehicle at each timestamp
    """

    Distance = np.zeros(len(journey))
    Slope = np.zeros(len(journey))
    l_route = journey.shape[0]

    for i in range(0, l_route - 1):  # end before i + 1 out of range
        elev_change = journey.Altitude.iloc[i+1] - journey.Altitude.iloc[i]
        # elev_change = journey['ElevChange'][i]
        dist_lateral = geopy.distance.geodesic(
            journey.Coordinates.iloc[i],
                # Lateral distance in meters - dist between two lat/lon coord
                # pairs
            journey.Coordinates.iloc[i+1]).m
                # Coordinates is
                # list(zip(journey['Latitude'], journey['Longitude']))

        dist_3d = np.sqrt(dist_lateral**2 + elev_change**2)
            # geodesic distance (3d = accounting for elevation), in meters
        if i == 0:
            Distance[i] = 0
        else:
            Distance[i] = dist_3d
        if Distance[i] != 0 and elev_change != 0:
            Slope[i] = np.arcsin(elev_change/dist_3d)
                # calculate slope angle in radians from opposite/hypotenuse

    journey['Displacement_m'] = list(Distance)  # add displacement to dataframe
    journey['slope_rad'] = list(Slope)  # add slope in radians to dataframe
    journey['slope_deg'] = journey['slope_rad'] * 180/np.pi
        # calculate slope angle in degrees


# -----------------------------------------------------------------------------
# Functions that calculate three environmental forces acting on
# the vehicle in N.

def getRoadFriction(mass, c_rr, slope, vel, grav=9.81):
    """ Road load (friction) (N)
        Inputs:
            c_rr is coeff of rolling resistance """
    rf = 0
    if vel > 0.3:
        rf = -mass * grav * c_rr * np.cos(slope)
    return rf


def getAerodynamicDrag(c_d, A, vel, rho=1.184):
    """ Aerodynamic Drag Force (N)
        Inputs:
           rho is air density 20C
           c_d is air drag coeff """
    return -0.50 * rho * c_d * A * vel**2


# -----------------------------------------------------------------------------
def getRoadSlopeDrag(mass, slope, grav=9.81):
    """ Road Slope Force (N) """
    return -mass * grav * np.sin(slope)


class Drivecycle:
    """
    Inputs: dataframe with journey info
    Outputs: drivecycle class
    """

    def __init__(self, journey):
        self.displacement = journey.Displacement_m  # m
        self.velocity = journey.Velocity  # m/s
        self.slope = journey.slope_rad  # rad
        self.time = journey.RelTime.max()  # Total Time Elapsed
        self.dt = journey.DeltaT  # Time elapsed between each timestamp
        self.dv = journey.DeltaV  # Difference in velocity between two consectuive timesteps
        self.acceleration = journey.Acceleration  # dv/dt


# -----------------------------------------------------------------------------
class Vehicle:
    """
    Inputs: Physical parameters of vehicle for modeling energy consumption
    Returns an array of vehicle energy consumption at each timestamp

    mass - vehicle mass (kg)
    cd - coefficient of drag
    crr - coefficient of rolling resistance
    A - vehicle frontal area (m^2)
    eff - vehicle propulsion efficiency
    rgbeff - regenerative braking energy recuperation efficiency
    cap - vehicle battery cap (kWh)
    p0 - constant power intake (W)
    """

    def __init__(self, mass=3900, payload=0, cd=0.36, crr=0.02, A=4,
                 eff=0.9, rgbeff=0.65, cap=100, p0=100):

        # TODO Read these parameters from the typ.xml file.

        # Vehicle physical parameters
        self.mass = mass  # kg
        self.load = payload  # kg
        self.crr = crr  # coefficient of rolling resistance
        self.cd = cd  # air drag coefficient
        self.A = A  # m^2, Approximation of vehicle frontal area
        self.eff = eff  # %, powertrain efficiency
        self.rgbeff = rgbeff  # %, regen brake efficiency
        # self.capacity = cap  # not used
        # self.battery = cap # not used
        self.p0 = p0  # constant power loss in W (to run the vehicle besides driving)

    def getEnergyExpenditure(self,cycle,regbrake=True):
        # computes energy expenditure from a Drivecycle object
        # dt default 1 second

        v = cycle.velocity  # m/s
        s = cycle.slope  # rad
        a = cycle.acceleration  # m/s^2
        dt = cycle.dt  # s
        # d = cycle.displacement  # m  [Not used.]
        dv = cycle.dv  # m/s

        # TODO Do this in the init.
        if regbrake:
            RGBeff = self.rgbeff  # static regen coeff
        else:
            RGBeff = 0

        # ---------------------------------------------------------------------

        Er = []  # Total energy consumption (J), from battery if positive, to
                 # battery if negative (RGbrake)

        Frpb = []  # force propulsive or braking

        # Drag forces
        Fa = []  # aerodynamic drag
        Frr = []  # rolling resistance
        Fhc = []  # hill climb (slope drag)
        Fr = []  # Drag force: Inertia + Friction + Aero Drag + Slope Drag (N)

        # ---------------------------------------------------------------------

        for slope,vel,acc,delta_t,delta_v in zip(s,v,a,dt, dv):

            if vel == 0:
                force, frr, fa, fhc = 0,0,0,0
            else:
                frr = getRoadFriction(self.mass,self.crr, slope, vel)
                fa = getAerodynamicDrag(self.cd, self.A, vel)
                fhc = getRoadSlopeDrag(self.mass, slope)

            Frr.append(frr)
            Fa.append(fa)
            Fhc.append(fhc)

            force = frr + fa + fhc  # (N + N + N)  - total drag force

            exp_speed_delta = force * delta_t / self.mass  # (N) * (s) / (kg)

            unexp_speed_delta = delta_v - exp_speed_delta  # (m/s) - (m/s)

            try:
                prop_brake_force = unexp_speed_delta / delta_t * self.mass
                    # (m/s) / (s) * (kg) = N

                kinetic_power = prop_brake_force * vel  # (N) * (m/s) = (W)

                propultion_work = kinetic_power * delta_t
                    # (W) * (s) -- kinetic energy

            except ZeroDivisionError:
                prop_brake_force = 0
                kinetic_power = 0
                propultion_work = 0

            # -----------------------------------------------------------------

            Fr.append(force)  # N
            Frpb.append(prop_brake_force)  # N
            Er.append(propultion_work)  # Ws

        ErP = [0.0]*len(Er)
        ErB = [0.0]*len(Er)
        offtake_power = [0.0]*len(Er)

        for i in range(len(Er)):
            offtake_power[i] = self.p0  # constant offtake

            if Er[i] > 0:  # energy that is used for propulsion
                ErP[i] = Er[i]/self.eff

            elif Er[i] < 0:  # energy that is regen'd back into the battery
                ErB[i] = Er[i] * RGBeff

        # TODO Return this as a dictionary, rather than ordered values.
        return Fa, Frr, Fhc, Fr, Frpb, Er, ErP, ErB, offtake_power
            # N,N,N,N,N,m/s,ms,N,Ws, Ws,Ws,Ws


def simulate():
    ...
