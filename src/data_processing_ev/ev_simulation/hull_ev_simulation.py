"""This function intakes a pandas dataframe of trip data ('journey'), and adds
columns with the geodesic distance and slope angles between consecutive
observations"""

INTEGRATION_MTHD = {'fwd': 0, 'bwd': 1, 'ctr': 2}
DFLT_INTEGRATION_MTHD = INTEGRATION_MTHD['bwd']
GRAV = 9.80665


import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as et
import ast
import data_processing_ev as dpr
from data_processing_ev.results_analysis import analysis_functions
import math
import multiprocessing as mp
from itertools import repeat


def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    # TODO Add progress bars:
    # https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def tryeval(val):
    try:
        val = ast.literal_eval(val)
    except ValueError:
        if val == 'true': val = True
        if val == 'false': val = False
        pass
    return val


def delta_fwd(array: pd.Series):
    array_shifted_bwd = array.shift(-1)
    array_cpy = pd.Series.copy(array)
    return pd.Series(array_shifted_bwd - array_cpy, name=array.name)


def delta_bwd(array: pd.Series):
    array_shifted_fwd = array.shift(1)
    array_cpy = pd.Series.copy(array)
    return pd.Series(array_cpy - array_shifted_fwd, name=array.name)


def delta_ctr(array: pd.Series):
    array_shifted_fwd = np.roll(array, 1)
    array_shifted_fwd[0] = 0
    array_shifted_bwd = np.roll(array, -1)
    array_shifted_bwd[-1] = 0
    return pd.Series((array_shifted_bwd - array_shifted_fwd)/2,
                     name=array.name)


def correct_angles(angles: pd.Series):
    def _correct_angle(angle):
        while angle > math.pi:
            angle -= 2*math.pi
        while angle < -math.pi:
            angle += 2*math.pi
        return angle
    new_angles = angles.map(_correct_angle)
    return new_angles


# -----------------------------------------------------------------------------
def read_file(fcd_file: Path, geo: bool, integration_mthd=DFLT_INTEGRATION_MTHD, **kwargs):
    """ This function takes the filename of a GPS trip stored in a .csv file,
    and path to the file.

    It reads the file, cleans it, adds columns that are necessary for the
    kinetic model, and returns the file in a pandas dataframe. """

    # Load journey.
    fcd_file = dpr.decompress_file(fcd_file, **kwargs)
    journey = pd.read_csv(fcd_file)
    fcd_file = dpr.compress_file(fcd_file, **kwargs)

    # Set up dataframe for energy consumption estimations
    journey['Velocity'] = journey['vehicle_speed']
    journey.drop(columns=['vehicle_speed', 'vehicle_x', 'vehicle_y'])
    if 'vehicle_z' not in journey.columns:
        journey['vehicle_z'] = 0

    # Convert speed in km/h to m/s.
    # journey['Velocity'] = journey['Velocity']/3.6
    # OR: Observations less than 0.5 m/s are set to 0, due to GPS noise
    # journey['Velocity'] = np.where(journey['Velocity'] >= 0.5, journey['Velocity']/3.6, 0)

    if integration_mthd == INTEGRATION_MTHD['fwd']:
        # Calculate time between samples. Useful for kinetic model.
        journey['DeltaT'] = delta_fwd(journey['timestep_time'])
        # Calculate change in velocity between each timestep
        journey['DeltaV'] = delta_fwd(journey['Velocity'])
        # Change in heading between each timestep (in radians)
        journey['Deltaθ'] = delta_fwd(journey['vehicle_angle']) * np.pi / 180
        journey['Deltaθ'] = correct_angles(journey['Deltaθ'])

    elif integration_mthd == INTEGRATION_MTHD['bwd']:
        # Calculate time between samples. Useful for kinetic model.
        journey['DeltaT'] = delta_bwd(journey['timestep_time'])
        # Calculate change in velocity between each timestep
        journey['DeltaV'] = delta_bwd(journey['Velocity'])
        # Change in heading between each timestep (in radians)
        journey['Deltaθ'] = -delta_bwd(journey['vehicle_angle']) * np.pi / 180
        journey['Deltaθ'] = correct_angles(journey['Deltaθ'])

    elif integration_mthd == INTEGRATION_MTHD['ctr']:
        # Calculate time between samples. Useful for kinetic model.
        journey['DeltaT'] = delta_ctr(journey['timestep_time'])
        # Calculate change in velocity between each timestep
        journey['DeltaV'] = delta_ctr(journey['Velocity'])
        # Change in heading between each timestep (in radians)
        journey['Deltaθ'] = delta_ctr(journey['vehicle_angle']) * np.pi / 180
        journey['Deltaθ'] = correct_angles(journey['Deltaθ'])

    else:
        raise ValueError("Unknown integration method.")

    # Calculate acceleration
    journey['Acceleration'] = np.where(journey['DeltaT'] > 0, journey['DeltaV']/journey['DeltaT'], 0)

    # Joins lat/lon Coords into one column, useful for getDist function
    journey['Coordinates'] = list(zip(journey['vehicle_y'], journey['vehicle_x']))

    analysis_functions.getDistSlope(journey, integration_mthd, geo, **kwargs)

    return journey


# Functions that calculate three environmental forces acting on
# the vehicle in N.

def getRoadFriction(mass, c_rr, slope, vel, grav=GRAV):
    """ Road load (friction) (N)
        Inputs:
            c_rr is coeff of rolling resistance """
    # rf = 0
    # if vel > 0.3:
    #     rf = -mass * grav * c_rr * np.cos(slope)
    rf = -mass * grav * c_rr * np.cos(slope)
    return rf


def getAerodynamicDrag(c_d, A, vel, rho=1.2041): # rho=1.184):
    """ Aerodynamic Drag Force (N)
        Inputs:
           rho is air density 20C
           c_d is air drag coeff """
    return -0.50 * rho * c_d * A * vel**2


def getRoadSlopeDrag(mass, slope, grav=GRAV):
    """ Road Slope Force (N) """
    return -mass * grav * np.sin(slope)


def getRadialDrag(mass, c_rad, delta_θ, dist, vel):
    """ Radial drag (N) """
    # If the vehicle has gone further than 1 meter in one second, and the
    # vehicle has made a turn:
    if dist > 0.1 and delta_θ != 0:
        radius = dist/abs(delta_θ)
        # If the radius is tiny, limit F_rad so that it doesn't become too much:
        if radius < 0.0001:
            dpr.LOGGERS['main'].error("The turning radius is extremely tiny! "
                "There could be an issue with the input data...")
            radius = 0.0001
        # If the turning radius is large, ignore F_rad:
        elif radius > 10000:
            radius = None  # FIXME. This is 10000 in SUMO.
        if radius is not None:
            F_rad = -c_rad * mass * vel**2 / radius
        else:
            F_rad = 0
    else:
        F_rad = 0
    return F_rad


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

    def __init__(self, scenario_dir, journey: pd.DataFrame,
                 name: str = 'unknown_vehicle', **kwargs):
        """
        Optional keyworded arguments:
            internalMomentOfInertia_is_mass:
                If this is true, the units of the internalMomentOfInertia will
                be taken as kg, instead of kg·m^2. Therefore, it will be
                multiplied by the radius of the wheel squared.
        """

        self.journey = journey

        config_xml = et.parse(next(scenario_dir.joinpath('_Inputs', 'Configs').glob('*vtype.xml')))
        params_xml = config_xml.getroot().find('vType').findall('param')
        params = {}

        for param in params_xml:
            key = param.get('key')
            val = param.get('value')
            val = tryeval(val)
            params[key] = val

        # Vehicle physical parameters
        self.mass = params['vehicleMass']  # kg
        self.crr = params['rollDragCoefficient']  # coefficient of rolling resistance
        self.cd = params['airDragCoefficient']  # air drag coefficient
        self.A = params['frontSurfaceArea']  # m^2, Approximation of vehicle frontal area
        self.eff = params['propulsionEfficiency']  # %, powertrain efficiency
        self.rgbeff = params['recuperationEfficiency']  # %, regen brake efficiency
        self.capacity = params['maximumBatteryCapacity']  # The total battery capacity (Wh)
        self.max_power = params['maximumPower']  # The maximum power that the vehicle can output.
        self.battery = self.capacity / 2  # Initial battery energy (Wh)
        self.p0 = params['constantPowerIntake']  # constant power loss in W (to run the vehicle besides driving)
        if 'wheelRadius' in params.keys():
            self.r_wheel = params['wheelRadius']
        else:
            self.r_wheel = 0.68 / 2
        if not kwargs.get('internalMomentOfInertia_is_mass', True):
            self.Inertia = params['internalMomentOfInertia']  # Internal moment of inertia was in kg·m2.
        else:
            self.Inertia = params['internalMomentOfInertia'] * self.r_wheel**2  # Internal moment of inertia was in kg·m4.

        self.c_rad = params['radialDragCoefficient']  # Radial drag coefficient.

        self.name = name

    def getEnergyExpenditure(self) -> pd.DataFrame:

        # computes energy expenditure from the journey dataframe

        journey = self.journey

        timesteps = journey['timestep_time']
        v = journey['Velocity']  # m/s
        s = journey['slope_rad']  # rad
        a = journey['Acceleration']  # m/s^2
        dt = journey['DeltaT']  # s
        dv = journey['DeltaV']  # m/s
        dθ = journey['Deltaθ']  # rad

        # TODO Do this in the init.
        RGBeff = self.rgbeff  # static regen coeff

        # ---------------------------------------------------------------------

        Er = []  # Total energy consumption (J), from battery if positive, to
                 # battery if negative (RGbrake)

        Frpb = []  # force propulsive or braking

        # Battery state
        bat_state = []  # The battery state for each timestep.

        # ---------------------------------------------------------------------
        for timestep, slope,vel,acc,delta_t,delta_v,delta_θ in zip(timesteps, s, v, a, dt, dv, dθ):

            prev_vel = vel - acc * delta_t
            dist = vel*delta_t

            if vel == 0:
                Frpb.append(0)
                Er.append(0)
            else:
                # TODO The below 3 drag forces use the current velocity as a
                # parameter. Change this to the average velocity! Also in SUMO.
                Err = -getRoadFriction(self.mass,self.crr, slope, vel) * dist
                Ea = -getAerodynamicDrag(self.cd, self.A, vel) * dist
                Erd = -getRadialDrag(self.mass, self.c_rad, delta_θ, dist, vel) * dist
                Ehc = -getRoadSlopeDrag(self.mass, slope) * dist
                Ek = 1/2 * self.mass * (vel**2 - prev_vel**2)
                Ek_rot = 1/2 * self.Inertia / self.r_wheel**2 * (vel**2 - prev_vel**2)
                Econst = self.p0

                E_net = Err + Ea + Ehc + Erd + Ek + Ek_rot + Econst
                F_net = E_net / dist

                if E_net/delta_t > self.max_power:
                    dpr.LOGGERS['main'].error(
                        f"Vehicle {self.name}: Timestep {timestep}: "
                        f"Vehicle required a power of {E_net/delta_t} W which "
                        f"is greater than the maximum power {self.max_power} W "
                        "that the vehicle is able to output.")

                Frpb.append(F_net)  # N
                Er.append(E_net)  # Ws

        Er = [Er_i / (3600) for Er_i in Er]  # Converting Ws to Wh
        Er_batt = [0.0]*len(Er)

        # TODO
        # Er_charged = [0.0]*len(Er)  # Energy charged from charging stations.
                                      # Not yet implemented! Haven't added
                                      # charging stations to SUMO yet. Perhaps,
                                      # if Chris Hull does that, we can implement
                                      # it here too.

        cur_bat_state = self.battery  # Current battery state.
        for i in range(len(Er)):
            if Er[i] >= 0:  # energy that is used for propulsion
                Er_batt[i] = Er[i]/self.eff

            else:  # energy that is regen'd back into the battery
                Er_batt[i] = Er[i] * RGBeff

            # # FIXME SUMO adds constant consumers as part of the propulsion work.

            # # If the vehicle is not stationary, we add the energy drawn by
            # # constant power auxiliary loads.
            # if v[i] != 0:
            #     Er_batt[i] += dt[i] * self.p0 / 3600  # Ws to Wh

            if np.isnan(Er_batt[i]):
                Er_batt[i] = 0

            cur_bat_state -= Er_batt[i]

            bat_state.append(cur_bat_state)

        self.battery_output = pd.DataFrame([
                journey['timestep_time'].values,
                journey['Acceleration'],
                bat_state,
                # Er_charged,
                Er_batt,
                # journey['vehicle_id'],
                # journey['vehicle_lane'],
                [self.capacity]*len(Er),
                # journey['vehicle_pos'],
                journey['displacement'],
                journey['Velocity'],
                journey['vehicle_x'],
                journey['vehicle_y'],
            ]).T
        self.battery_output.columns = [
            'timestep_time',
            'vehicle_acceleration',
            'vehicle_actualBatteryCapacity',
            # 'vehicle_energyCharged',
            'vehicle_energyConsumed',
            # 'vehicle_id',
            # 'vehicle_lane',
            'vehicle_maximumBatteryCapacity',
            # 'vehicle_posOnLane',
            'displacement',
            'vehicle_speed',
            'longitude',
            'latitude'
        ]

        return self.battery_output


def simulate_trace(scenario_dir: Path, fcd_file: Path, geo: bool,
        integration_mthd=DFLT_INTEGRATION_MTHD, **kwargs) -> pd.DataFrame:
    """Simulates the Hull model on the one FCD trace."""

    # Read data
    journey = read_file(fcd_file, geo, integration_mthd, **kwargs)

    # Initialise vehicle
    vehicle = Vehicle(scenario_dir, journey, name=' - '.join([fcd_file.parents[1].name, fcd_file.parents[0].name]), **kwargs)

    # Execute EV simulation
    battery_output = vehicle.getEnergyExpenditure()

    return battery_output


def _check_if_geo_inputs(scenario_dir: Path) -> bool:
    config_file = next(scenario_dir.joinpath('EV_Simulation', 'Sumocfgs_Combined').glob('*.sumocfg'))
    config = et.parse(config_file)
    try:
        geo = config.getroot().find('output').find('fcd-output.geo').get('value')
        geo = True if geo == "true" else False

    except AttributeError:
        geo = False

    return geo


def simulate(scenario_dir: Path,
        integration_mthd=DFLT_INTEGRATION_MTHD,
        routing_was_done=False, **kwargs):

    fcd_files = sorted([*scenario_dir.joinpath('Mobility_Simulation', 'FCD_Data').\
        glob('*/*/fcd.out.csv*')])

    if routing_was_done:
        # If we used SUMO's routing and mobility simulation to generate the FCD
        # data, check if the coordinates are geographical (as opposed to
        # cartesian).
        geo = _check_if_geo_inputs(scenario_dir)
    else:
        # We assume that the FCD-converted 1 Hz data contains geographical
        # coordinates.
        geo = True

    battery_outputs = []

    args_simulation = zip(
        repeat(scenario_dir),
        fcd_files,
        repeat(geo),
        repeat(integration_mthd))
    kwargs_simulation = repeat(kwargs)

    with mp.Pool(mp.cpu_count() - 1) as p:
        battery_outputs = starmap_with_kwargs(p, simulate_trace,
            args_simulation, kwargs_simulation)
    # # OR Old multithreaded way
    # for fcd_file in tqdm(fcd_files):
    #     battery_output = simulate_trace(scenario_dir, fcd_file, geo, integration_mthd, **kwargs)
    #     battery_outputs.append(battery_output)

    for fcd_file, battery_output in tqdm(zip(fcd_files, battery_outputs)):

        # Write results
        ev_name = fcd_file.parents[1].name
        date = fcd_file.parent.name
        output_file = scenario_dir.joinpath(
            'EV_Simulation', 'EV_Simulation_Outputs',
            ev_name, date, 'battery.out.csv')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        battery_output.to_csv(output_file, index=False)
        output_file = dpr.compress_file(output_file, **kwargs)

    print("EV Simulation completed. Basic stats:")
    energy_consumption = []
    for journey_df in battery_outputs:
        dist = journey_df['displacement'].sum()
        # Alternative way of calculating dist:
        # t = journey_df['timestep_time'] - journey_df['timestep_time'][0]
        # dist = integrate.cumtrapz(journey_df['vehicle_speed'], t)[-1]
        energy_consumption.append(journey_df['vehicle_energyConsumed'].sum()/dist)

    summ_df = pd.DataFrame({'Energy efficiency (kWh/km)': pd.Series(energy_consumption)})

    print(round(summ_df.describe(),2))
    summ_df.boxplot(showmeans=True)
