#!/usr/bin/env python3

"""
This script generates box-plots of PV charging potential. It takes the stop-
arrival and -duration data from the Time Clustering step. It then ...
"""
# FIXME Incorporate this into "Results Analysis" module of code repository.

import os
from pathlib import Path
import pandas as pd
from scipy import integrate, interpolate
import typing as tp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle


def _load_irradiance_profile(scenario_dir: Path) -> tp.Callable:
    # Load irradiance profile as a DataFrame.
    irradiance_file = [*scenario_dir.joinpath('SAM_Simulation', 'Results',
                                              'POA_Irradiance_Profile', 'Data'
                                              ).glob('*.csv')][0]
        # TODO Don't rely on just one year, maybe take the average across
        # multiple years... And maybe keep each month seperate.
    irradiance_df = pd.read_csv(irradiance_file)
    irradiance_df.columns = ['Time', 'Irradiance']

    # Apply efficiency constant, to get PV output profile.
    eff = 0.80 * 0.20
    pv_output_df = irradiance_df  # Make a copy of irradiance_df.
    pv_output_df['Irradiance'] *= eff  # Multiply irradiance by efficiency
                                       # constant.

    # FIXME Delete this line and make sure that it hasn't introduced bugs.
    # pv_output_df['Irradiance'] /= 1000  # Convert from W/m^2 to kW/m^2

    pv_output_df.columns = ['Time', 'P_Out']

    # Create a function which interpolates between points of pv_output_df. This
        # will be used for integration.
    f_pv_out = interpolate.interp1d(
        x=pv_output_df['Time'], y=pv_output_df['P_Out'],
        fill_value='extrapolate'
    )

    def f_pv_out_multiple(time):
        if time >= 24:
            while time >= 24:
                time -= 24
        return f_pv_out(time)

    return f_pv_out_multiple

def dirty__plot_stop_duration_profile(stops_df: pd.DataFrame) -> None:
    stops_df['Arrival_Hour'] = np.digitize(
            stops_df['Stop_Arrival'], range(1, 24), right=True)
    stops_df = stops_df.reset_index().\
            set_index(['EV_Name', 'Date', 'Arrival_Hour'])
    stop_duration_profile = pd.DataFrame()
    stop_duration_profile['Mean'] = stops_df.groupby('Arrival_Hour').mean()['Stop_Duration']
    stop_duration_profile['Std'] = stops_df.groupby('Arrival_Hour').std()['Stop_Duration']

    plt.errorbar(stop_duration_profile.index, stop_duration_profile['Mean'],
                 stop_duration_profile['Std'], ecolor='orange', linestyle=None, marker='o')
    plt.ylabel("Mean stop duration")
    plt.xlabel("Stop arrival time")

    fig_dir = scenario_dir.joinpath("SAM_Simulation",
        "dirty__stop_duration_profile.fig.pickle")  # XXX Wrong save_path.
    fig = plt.gcf()
    pickle.dump(fig, open(fig_dir, 'wb'))
    fig_dir = scenario_dir.joinpath("SAM_Simulation",
        "dirty__stop_duration_profile.pdf")
    plt.savefig(fig_dir)

    plt.show()
    return None

def main(scenario_dir: Path):

    # Load irradiance profile
    f_pv_out = _load_irradiance_profile(scenario_dir)

    # Load list of stop-arrivals and -durations as DataFrame.
    stops_file = scenario_dir.joinpath('Temporal_Clusters', 'Clustered_Data',
                                       'dirty__time_clusters_with_dates.csv')
    stops_df = pd.read_csv(stops_file).set_index(['EV_Name', 'Date'])
    dirty__plot_stop_duration_profile(stops_df)
        # TODO convert dates to `datetime.date`s.

    # # Load daily energy-consumption dataframe (from SUMO simulation-outputs).
    # energy_usage_file = scenario_dir.joinpath(
    #     'Simulations', 'Simulation_Outputs', 'Energy_usage.csv'
    # )
    # energy_balance_df = pd.read_csv(energy_usage_file)
    # energy_balance_df.columns = ['EV_Name', 'Date', 'Energy_Used']
    #     # TODO Fix the column names in the original source code, rather.
    # energy_balance_df = energy_balance_df.set_index(['EV_Name', 'Date'])

    # For each stop-event, calculate the PV energy that could be used for
        # charging during the stop-event.
    # -------------------------------------------------------------------

    pv_potentials = []
    integration_errors = []
    for _, stop in tqdm(stops_df.iterrows()):
        # Get the arrival and departure times of the stop-event.
        stop_beg = stop['Stop_Arrival']
        stop_end = stop_beg + stop['Stop_Duration']
        # Integrate to find the PV energy for this stop-event.
        stop_pv_potential, error = integrate.quad(f_pv_out, stop_beg, stop_end)
        integration_errors.append(error)
        # Create a copy of the stop
        stop_copy = stop.copy()
        stop_copy['PV_Charge_Pot_Per_m2'] = stop_pv_potential
        pv_potentials.append(stop_copy)
    pv_potentials_df = pd.DataFrame(pv_potentials)
    pv_potentials_df.index = pd.MultiIndex.from_tuples(pv_potentials_df.index)
    pv_potentials_df.index.names = ['EV_Name', 'Date']

    # For each date, sum up the total charging potential for that date.
    pv_potentials_df = pd.DataFrame(
        pv_potentials_df.groupby(
            ['EV_Name', 'Date']
        )['PV_Charge_Pot_Per_m2'].sum()
    )

    # For each date, subtract the EV energy consumption from the charging
        # potential (assuming a certain PV size in m^2) to get the
        # "charging potential balance".
    # TODO. Assume a m^2 PV area, and multiply that by the "per-m2" PV charging
        # potential, to create a new column. Then subtract this column from the
        # "Energy Usage" column, to find the balance.

    # Generate box plots of the charging potential per m2.
    # ----------------------------------------------------

    # Seperate dataframe into multiple, based on the ev_name.
    all_pv_charge_potentials = [
        pv_potentials_df.loc[ev_name]['PV_Charge_Pot_Per_m2'] for ev_name in
        pv_potentials_df.index.get_level_values('EV_Name').unique()
    ]
    # Extract a list of ev name.
    ev_names = list(
        pv_potentials_df.index.get_level_values('EV_Name').unique()
    )

    # Generate plot
    plt.boxplot([pv_charge_potentials/1000 for pv_charge_potentials in
                 all_pv_charge_potentials])  # Diving by 1000 to convert from
                                             # Wh to kWh.
    plt.title("Daily PV charging potential per EV")
    plt.ylabel("Daily PV charging potential (kWh/m^2)")
    plt.xticks(range(1, len(ev_names)+1), ev_names)
    plt.xlabel("EV ID")

    for i, ev_df in enumerate(all_pv_charge_potentials):
        # Generate random x-values centered around the box-plot.
        x = np.random.normal(loc=1+i, scale=0.04, size=len(ev_df))
        plt.scatter(x, ev_df, alpha=0.4)

    # Generate box-plots of the charging potential balance.
    # TODO

    # Save the box-plots and dataframes.
    save_dir = scenario_dir.joinpath('SAM_Simulation')
    # Output box-plots.
    # As png:
    fig_dir = save_dir.joinpath("dirty__pv_charging_pot_box_plots.png")
    plt.savefig(fig_dir)
    # As svg:
    fig_dir = save_dir.joinpath("dirty__pv_charging_pot_box_plots.svg")
    plt.savefig(fig_dir)
    # As pickle:
    fig_dir = save_dir.joinpath(
        "dirty__pv_charging_pot_box_plots.fig.pickle")
    fig = plt.gcf()
    pickle.dump(fig, open(fig_dir, 'wb'))

    pv_potentials_df.to_csv(
        save_dir.joinpath('dirty__pv_charging_pot_df.csv')
    )

    plt.show()


if __name__ == "__main__":
    scenario_dir = Path(os.path.abspath(__file__)).parents[1]
    main(scenario_dir)
