"""This function intakes a pandas dataframe of trip data ('journey'), and adds
columns with the geodesic distance and slope angles between consecutive
observations"""

import pandas as pd
import numpy as np
import geopy.distance
from pathlib import Path
from tqdm import tqdm


def read_file(fcd_file: Path):
    """ This function takes the filename of a GPS trip stored in a .csv file,
    and path to the file.

    It reads the file, cleans it, adds columns that are necessary for the
    kinetic model, and returns the file in a pandas dataframe. """

    # Load journey.
    journey = pd.read_csv(fcd_file)

    # Set up dataframe for energy consumption estimations
    journey['Velocity'] = journey['vehicle_speed']
    journey['Longitude'] = journey['vehicle_x']
    journey['Latitude'] = journey['vehicle_y']
    journey['Altitude'] = journey['vehicle_z']

    journey.drop(columns=['vehicle_speed', 'vehicle_x', 'vehicle_y',
                          'vehicle_z'])

    # Convert speed in km/h to m/s.
    # journey['Velocity'] = journey['Velocity']/3.6
    # OR: Observations less than 0.5 m/s are set to 0, due to GPS noise
    # journey['Velocity'] = np.where(journey['Velocity'] >= 0.5, journey['Velocity']/3.6, 0)

    # Calculate elevation change
    elev_change = journey['Altitude'] - journey['Altitude'].shift(1)
    journey['ElevChange'] = np.where(abs(elev_change) >= 0.2, elev_change, 0)

    # Calculate time between samples. Useful for kinetic model.
    journey['DeltaT'] = journey['timestep_time'] - \
                        journey['timestep_time'].shift(1)

    # Calculate change in velocity between each timestep
    journey['DeltaV'] = journey['Velocity'] - \
                        journey['Velocity'].shift(1)

    # Calculate acceleration
    journey['Acceleration'] = np.where(journey['DeltaT'] > 0, journey['DeltaV']/journey['DeltaT'], 0)

    # Joins lat/lon Coords into one column, useful for getDist function
    journey['Coordinates'] = list(zip(journey['Latitude'], journey['Longitude']))

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
    cap - vehicle battery cap (Wh)
    p0 - constant power intake (W)
    """

    # mass=3900, c_d=0.36, c_rr=0.02, A=4, propulsion_eff=0.9, regen_eff=0.65, constant_power=100
    # def __init__(self, journey: pd.DataFrame, mass=3900, payload=0, cd=0.36,
    #              crr=0.02, A=4, eff=0.9, rgbeff=0.65, cap=1000000, p0=100,
    #              regbrake=True):

    # mass=2900, c_d=0.35, c_rr=0.01, A=4, propulsion_eff=0.8, regen_eff=0.5, constant_power=100
    def __init__(self, journey: pd.DataFrame, mass=2900, payload=0, cd=0.35,
                 crr=0.01, A=4, eff=0.8, rgbeff=0.5, cap=2000000,
                 ini_cap=1000000, p0=100, regbrake=True):

        # TODO: Doesn't implement InternalMomentOfInertia, radialDragCoefficient

        self.journey = journey
        getDistSlope(self.journey)

        self.regbrake = regbrake

        # TODO Read these parameters from the typ.xml file. TODO

        # Vehicle physical parameters
        self.mass = mass  # kg
        self.load = payload  # kg (TODO Not yet implemented.)
        self.crr = crr  # coefficient of rolling resistance
        self.cd = cd  # air drag coefficient
        self.A = A  # m^2, Approximation of vehicle frontal area
        self.eff = eff  # %, powertrain efficiency
        self.rgbeff = rgbeff  # %, regen brake efficiency
        self.capacity = cap  # The total battery capacity (Wh)
        self.battery = cap  # Initial battery energy (Wh)
        self.p0 = p0  # constant power loss in W (to run the vehicle besides driving)

    def getEnergyExpenditure(self) -> pd.DataFrame:

        # computes energy expenditure from the journey dataframe

        journey = self.journey
        regbrake = self.regbrake

        v = journey['Velocity']  # m/s
        s = journey['slope_rad']  # rad
        a = journey['Acceleration']  # m/s^2
        dt = journey['DeltaT']  # s
        dv = journey['DeltaV']  # m/s

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

        # Battery state
        bat_state = []  # The battery state for each timestep.

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

                propulsion_work = kinetic_power * delta_t
                    # (W) * (s) -- kinetic energy

                # TODO: No internal moment of inertia used to calculate
                # propulsion work...

            except ZeroDivisionError:
                prop_brake_force = 0
                kinetic_power = 0
                propulsion_work = 0

            # -----------------------------------------------------------------

            Fr.append(force)  # N
            Frpb.append(prop_brake_force)  # N
            Er.append(propulsion_work)  # Ws

        Er = [Er_i / (3600) for Er_i in Er]  # Converting Ws to Wh
        Er_batt = [0.0]*len(Er)
        Er_charged = [0.0]*len(Er)  # Energy charged from charging stations.
                                    # Not yet implemented! Haven't added
                                    # charging stations to SUMO yet. Perhaps,
                                    # if Chris Hull does that, we can implement
                                    # it here too.
        offtake_power = [0.0]*len(Er)

        cur_bat_state = self.battery  # Current battery state.
        for i in range(len(Er)):
            offtake_power[i] = self.p0  # constant offtake

            if Er[i] > 0:  # energy that is used for propulsion
                Er_batt[i] = Er[i]/self.eff

            elif Er[i] < 0:  # energy that is regen'd back into the battery
                Er_batt[i] = Er[i] * RGBeff

            cur_bat_state -= Er_batt[i]

            bat_state.append(cur_bat_state)

        self.battery_output = pd.DataFrame([
                journey['timestep_time'].values,
                journey['Acceleration'],
                bat_state,
                Er_charged,
                Er_batt,
                journey['vehicle_id'],
                journey['vehicle_lane'],
                [self.capacity]*len(Er),
                journey['vehicle_pos'],
                journey['Velocity'],
                journey['Longitude'],
                journey['Latitude'],
            ]).T
        self.battery_output.columns = [
            'timestep_time',
            'vehicle_acceleration',
            'vehicle_actualBatteryCapacity',
            'vehicle_energyCharged',
            'vehicle_energyConsumed',
            'vehicle_id',
            'vehicle_lane',
            'vehicle_maximumBatteryCapacity',
            'vehicle_posOnLane',
            'vehicle_speed',
            'longitude',
            'latitude'
        ]

        # TODO Add to a dictionary and save as property.
        Fa, Frr, Fhc, Fr, Frpb, Er, Er_batt, offtake_power
            # N,N,N,N,N,m/s,ms,N,Wh,Wh,Wh,Wh

        return self.battery_output


def simulate(scenario_dir: Path, **kwargs):

    fcd_files = sorted([*scenario_dir.joinpath('EV_Simulation',
        'SUMO_Simulation_Outputs').glob('*/*/fcd.out.csv*')])

    for fcd_file in tqdm(fcd_files):
        # Read data
        journey = read_file(fcd_file)

        # Initialise vehicle
        vehicle = Vehicle(journey)

        # Execute EV simulation
        battery_output = vehicle.getEnergyExpenditure()

        # Write results
        ev_name = fcd_file.parents[1].name
        date = fcd_file.parent.name
        output_file = scenario_dir.joinpath(
            'EV_Simulation', 'Hull_Simulation_Outputs',
            ev_name, date, 'battery.out.csv')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        battery_output.to_csv(output_file, index=False)
