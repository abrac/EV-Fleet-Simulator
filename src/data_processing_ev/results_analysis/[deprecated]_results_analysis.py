# TODO:
# - [ ] Refactor "indiv_ev" and "indiv_vehicle" to ev
# - [ ] Refactor "agg_ev" and "agg_vehicle" to fleet
# - [ ] Save pickled versions of plots

# %% Imports ##################################################################

import os
import sys
import itertools  # noqa
import pandas as pd
import numpy as np  # noqa
from pathlib import Path
import subprocess
import logging
import typing as typ
from tqdm import tqdm
import scipy.integrate as integrate
import datetime as dt
import json
import mpl_axes_aligner.align as mpl_align
import statistics
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt  # noqa

if "SUMO_HOME" in os.environ:
    xml2csv = Path(os.environ["SUMO_HOME"], "tools", "xml", "xml2csv.py")
else:
    sys.exit("Please declare environmental variable 'SUMO_HOME'.")

# plt.ion()
plt.style.use('default')


# %% Analysis Class ###########################################################

class Data_Analysis:
    def __create_csvs(self, ev_sim_dirs: typ.Sequence[Path]) -> None:
        """Convert all battery.out.xml files to csv files and save them

        They will be saved in a corresponding folder in
        {Scenario_Dir}/Results/
        """

        # TODO Put this in main.py
        _ = input("Would you like to (re)convert all " +
                  "battery.out.xml files to csv? y/[n] \n\t")
        convert = (True if _.lower() == 'y' else False)
        if convert:
            print("\nConverting xml files to csv...")
            for ev_sim_dir in tqdm(ev_sim_dirs):
                # Try create ev_csv if it doesn't exist
                ev_name = ev_sim_dir.parents[0].name
                # FIXME FIXME Replace the below statement with:
                # date = ev_sim_dir.name
                date = '_'.join(ev_sim_dir.name.split('_')[1:])
                battery_csv = self.__scenario_dir.joinpath(
                    'Results', ev_name, date, 'Battery.out.csv')
                battery_xml = ev_sim_dir.joinpath("Battery.out.xml")
                if not battery_xml.exists():
                    continue
                battery_csv.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(['python', xml2csv, '-o', battery_csv,
                                battery_xml])
                # Warn if battery_csv *still* doesn't exist
                if not battery_csv.exists():
                    logging.warning("Failed to create ev_csv in \n\t" +
                                    str(battery_csv.parent))

    def __init__(self, scenario_dir: Path,
                 ev_sim_dirs: typ.Sequence[Path]) -> None:
        """
        Create a data_analysis class.
        """
        self.__scenario_dir = scenario_dir
        # Convert all battery.out.xml files to csv files
        self.__create_csvs(ev_sim_dirs)
        self.__battery_csv_paths = [
            *scenario_dir.joinpath('Results').glob('*/*/Battery.out.csv')
        ]
        self.__agg_vehicle_dir = scenario_dir.joinpath('SUMO_Simulation',
                                                       'Simulation_Outputs')
        self.__indiv_vehicle_dirs = [*self.__agg_vehicle_dir.glob('*/')]

    def __ev_csv_to_df(self, ev_csv: Path, warn_nan: bool = False,
                       secs_to_dts: bool = False, delim=';') -> pd.DataFrame:
        """
        Inputs:
            secs_to_dts: True if function must convert seconds to datetime
                         objects in the 'timestep_time' column.
        """
        # Read the csv
        ev_df = pd.read_csv(ev_csv, sep=delim,
                            dtype={"vehicle_id": "category",
                                   "vehicle_lane": "category"})
        # If warn_nan flag is set, check if df has coordinates that are
        # NaN and delete them
        if warn_nan and True in ev_df['vehicle_x'].isna():
            logging.warning("NaN found in ev_df")
            ev_df = ev_df[ev_df['vehicle_x'].notna()]

        # Change times from seconds to date-time objects
        #   FIXME Don't use name from directory parent. Get from xml file.
        #   FIXME Only convert if the time column doesn't already consist of
        #         date-time objects. Currently adding a flag in function
        #         arguments...
        if (secs_to_dts):
            date_arr_str = ev_csv.parents[0].name.split('_')
            date_arr = [int(date_el) for date_el in date_arr_str]
            date = dt.date(*date_arr)
            ev_df['timestep_time'] = pd.to_datetime(
                ev_df['timestep_time'], unit='s', origin=date)
        else:
            ev_df['timestep_time'] = pd.to_datetime(ev_df['timestep_time'])
        return ev_df

    def __plot_multiple_lines(self, ev_dfs: typ.Iterable[pd.DataFrame],
                              plt_title: str, plt_type: str = 'Power',
                              ev_names: typ.List[str] = None,
                              MY_DPI: int = 96) -> plt.Figure:
        """
        Plot graphs of multiple battery output data for a given simulation.

        Inputs:
            ev_dfs List[pd.DataFrame]:
                Each DataFrame has a SUMO EV battery output. The battery output
                is in XML format, this must be converted to a CSV and then a
                Data-Frame.
            plt_type [str]:
                Type of graph is given by `plt_type`. Acceptable values are
                'Power', 'Speed' and 'Distance'.
            ev_names List[str]:
                Name of each EV corresponding to those in `ev_df`s. These are
                used for creating a plot legend.
        """
        fig = plt.figure(figsize=(800/MY_DPI, 600/MY_DPI), dpi=MY_DPI)

        # Create a dataframe `time` which spans the time of all ev_dfs
        #   Remove dates from the timestep_time column
        for ev_df in ev_dfs:
            ev_df['timestep_time'] = pd.to_datetime(
                ev_df['timestep_time'].dt.time,
                format='%H:%M:%S'
            )
        # TODO: Make `rolling_window` function argument
        rolling_window = 3600  # seconds

        # Get units for plot_type
        plt_units = {'Power': 'kW', 'Speed': 'km/h', 'Distance': 'km'}

        # Plot setup
        ax = fig.subplots()
        fig.suptitle(plt_title)
        ax.set_title(f'{plt_type} vs Time')
        ax.set_xlabel('Time')
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel(f'Rolling Average {plt_type} ' +
                      f'[{plt_units[plt_type]}]')
        ax.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter('%H:%M'))

        for ev_df in ev_dfs:
            if plt_type == 'Distance':
                # Approx_distance in km. Note: `ev_df['vehicle_speed']`
                # is in m/s.
                dist = integrate.cumtrapz(ev_df['vehicle_speed'], dx=1)

            # Get the column for plot_type
            plt_df_func = {
                'Power': lambda: ev_df['vehicle_energyConsumed']*3.6,
                'Speed': lambda: ev_df['vehicle_speed']*3.6,
                'Distance': lambda: pd.Series(dist/1000)}
            plt_df = plt_df_func[plt_type]()

            time = (ev_df['timestep_time'][1:] if plt_type == 'Distance' else
                    ev_df['timestep_time'])

            if plt_type in ('Power', 'Speed'):
                plt_df = plt_df.rolling(rolling_window, center=True).mean()

            # Plot
            ax.plot(time, plt_df)
        if ev_names:
            ax.legend(ev_names)
        ax.axhline(color="darkgrey")

        return fig

    def __plot_summary_graph(self, ev_df: pd.DataFrame, plt_title: str,
                             MY_DPI: int = 96) -> plt.Figure:
        """
        Plot the graph of battery output data for a given simulation.

        Inputs:
            ev_df pd.DataFrame:
                The DataFrame has a SUMO EV battery output. The battery output
                is in XML format, this must be converted to a CSV and then a
                Data-Frame.
        """
        # TODO Seperate each subplot into a new function.
        plt_fig = plt.figure(figsize=(800/MY_DPI, 600/MY_DPI), dpi=MY_DPI)

        time = ev_df['timestep_time']

        # Variables needed in plots
        # Approx_distance in km. Note: `ev_df['vehicle_speed']` is in m/s.
        dist = integrate.cumtrapz(ev_df['vehicle_speed'], dx=1)/1000
        # TODO: Make `rolling_window` function argument
        rolling_window = 3600  # seconds

        power_df = ev_df['vehicle_energyConsumed']*3.6  # 3.6 Wh/s in 1 kW
        power_df_rolling = ev_df['vehicle_energyConsumed'].rolling(
            rolling_window, center=True).mean()*3.6

        plt.suptitle(plt_title)

        ax_PvT = plt.subplot(321)
        ax_PvT.set_title('Power vs Time')
        ax_PvT.set_xlabel('Time')
        ax_PvT_Colour = 'tab:blue'
        ax_P2vT_Colour = 'tab:orange'
        # Plot instantaneous power
        ax_PvT.set_ylabel('Instantaneous Power (kW)', color=ax_PvT_Colour)
        ax_PvT.plot(time, power_df, color=ax_PvT_Colour)
        ax_PvT.axhline(color="darkgrey")
        plt.setp(ax_PvT.get_xticklabels(), rotation=45)
        # Plot rolling average power
        ax_P2vT = ax_PvT.twinx()
        ax_P2vT.set_ylabel('Rolling Average Power (kW)',
                           color=ax_P2vT_Colour)
        ax_P2vT.plot(time, power_df_rolling, color=ax_P2vT_Colour)
        mpl_align.yaxes(ax_PvT, 0, ax_P2vT, 0)
        ax_PvT.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter('%H:%M'))

        ax_PvX = plt.subplot(322)
        ax_PvX.set_title('Power vs Distance')
        ax_PvX.axhline(color="darkgrey")
        ax_PvX.set_xlabel('Distance (km)')
        ax_PvX_Colour = 'tab:blue'
        # Plot instantaneous power
        ax_PvX.set_ylabel('Instantaneous Power (kW)', color=ax_PvX_Colour)
        ax_PvX.plot(dist, power_df[1:], color=ax_PvX_Colour)
        ax_PvX.axhline(color="darkgrey")
        plt.setp(ax_PvX.get_xticklabels(), rotation=45)
        # # Plot rolling average power
        # #   TODO Make a dataframe that has rolling average of power with
        # # respect to distance
        # ax_P2vX_Colour = 'tab:orange'
        # ax_P2vX = ax_PvX.twinx()
        # ax_P2vX.set_ylabel('Rolling Average Power (kW)',
        #                    color=ax_P2vX_Colour)
        # ax_P2vX.plot(dist, power_df_rolling[1:], color=ax_P2vX_Colour)
        # mpl_align.yaxes(ax_PvX, 0, ax_P2vX, 0)

        plt.subplot(323)
        plt.title('Energy vs Time')
        plt.plot(time, 1000 - ev_df['vehicle_actualBatteryCapacity']/1000)
        plt.ylabel('Energy (kWh)')
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter('%H:%M'))

        plt.subplot(324)
        plt.title('Energy vs Distance')
        plt.plot(dist, 1000 - ev_df['vehicle_actualBatteryCapacity'][1:]/1000)
        plt.ylabel('Energy (kWh)')
        plt.xlabel('Distance (km)')

        plt.subplot(325)
        plt.title('Speed vs Time')
        # Note, vehicle_speed is m/s
        # ∴ x m/s = x / 1000 * 3600 km/h = x * 3.6 km/h = y km/h
        plt.plot(time, ev_df['vehicle_speed'] * 3.6)
        plt.ylabel(r'$ Speed\ (km \cdot h^{-1}) $')
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter('%H:%M'))

        plt.subplot(326)
        plt.title('Speed vs Distance')
        plt.plot(dist, ev_df['vehicle_speed'][1:] * 3.6)
        plt.ylabel(r'$ Speed\ (km \cdot h^{-1}) $')
        plt.xlabel('Distance (km)')

        plt.tight_layout()

        return plt_fig

    def make_plots(self) -> None:
        """
        Asks a few questions and generates plots accordingly

        Questions:
            - View?
                - All at once?
                - Save?
        """
        def plot_generator(paths: typ.Iterable[Path], save: bool = True
                           ) -> typ.Iterator[plt.Figure]:
            for ev_csv in tqdm(paths):
                # Get parent directory
                ev_dir = ev_csv.parent
                ev_df = self.__ev_csv_to_df(ev_csv, secs_to_dts=True)
                # Plot
                plt_fig = self.__plot_summary_graph(
                    ev_df, plt_title="Simulation Output of experiment: "
                    + f"{ev_dir.parents[1].name} > "
                    + f"{ev_dir.parents[0].name}")
                # If the user wants to save the plots
                if save:
                    # Create Graphs folder and save plot
                    graphsdir = ev_dir.joinpath('Graphs')
                    graphsdir.mkdir(parents=True, exist_ok=True)
                    save_path = graphsdir.joinpath('summary.svg')
                    plt_fig.savefig(save_path)
                yield(plt_fig)

        print(f"\nAbout to generate {len(self.__battery_csv_paths)} plots...")
        print("\nWould you like to view the plots? [y]/n")
        view_plots = False if input().lower() == 'n' else True
        if view_plots:
            print("\nWould you like to view all plots at once " +
                  "(rather than one-by-one)? [y]/n")
            blocking = True if input().lower() == 'n' else False
            if blocking:
                print("Close each plot so that the next one can appear.")
            print("\nWould you like to save the plots? [y]/n")
            saving = False if input().lower() == 'n' else True
        else:
            saving = True  # Save the plots if you're not viewing them.

        print("\nGenerating plots...")
        for fig in plot_generator(save=saving, paths=self.__battery_csv_paths):
            if view_plots and blocking:
                plt.show(block=True)
                plt.pause(10)
        if view_plots and not blocking:
            plt.show()

        input(f"\nPress Enter to {'close plots and ' if view_plots else ''}"
              + "continue program execution...")
        plt.close('all')

    def __gen_ev_dfs_of_ev(self, vehicle_dir: Path
                           ) -> typ.Iterator[pd.DataFrame]:
        """Generate dataframes of days that taxi has valid data."""
        for ev_day_dir in self.__battery_csv_paths:
            if vehicle_dir in ev_day_dir.parents:
                # Obtain a dataframe for day
                ev_df = self.__ev_csv_to_df(ev_day_dir,
                                            secs_to_dts=True).iloc[:-1]
                    # FIXME Why is final_energy NaN? Making it  # noqa
                    # 2nd last for now.                         # noqa
                # If ev_df ends on a different date than it started,
                # strip away excess rows.
                if (ev_df['timestep_time'].iloc[0].date() !=
                        ev_df['timestep_time'].iloc[-1].date()):
                    input("Warning! ev_df consists of " +
                          "more than one date. Press any " +
                          "key to delete excess rows.")
                    # Remove all rows with date after that of first row
                    #   Get index of first row with new date
                    first_row_date = ev_df[
                        'timestep_time'].iloc[0].date()
                    first_bad_idx = None
                    # FIXME This is inefficient!
                    for idx, row in ev_df.iterrows():
                        if (row['timestep_time'].date() >
                                first_row_date):
                            first_bad_idx = idx
                            break
                    ev_df = ev_df.iloc[:first_bad_idx]
                yield ev_df

    def save_ev_stats(self) -> None:
        print("\nGenerating individual ev stats...")
        save_plots = input("Would you like to save the plots? [y]/n \n\t")
        save_plots = False if save_plots.lower() == 'n' else True
        for vehicle_dir in tqdm(self.__indiv_vehicle_dirs):

            # Initialise a statistics tree
            stats_tree: typ.Dict = {
                "stats": {
                    "averages": None,
                    "dates": []
                }
            }

            # Get dataframes of each ev-day
            ev_dfs = [*self.__gen_ev_dfs_of_ev(vehicle_dir)]

            # Populate the statistics for the various dates
            for ev_day_df in ev_dfs:
                date = ev_day_df['timestep_time'][0].date()
                initial_time = ev_day_df['timestep_time'].iloc[0].time()
                final_time = ev_day_df['timestep_time'].iloc[-1].time()
                initial_energy = ev_day_df[
                    'vehicle_actualBatteryCapacity'].iloc[0]/1000
                final_energy = ev_day_df[
                    'vehicle_actualBatteryCapacity'].iloc[-1]/1000
                # `time_splits` which allow for arbitrary time-splitting
                # of statistics.
                time_splits_dict = ((6,0,0), (9,0,0), (12,0,0), (16,0,0),  # noqa
                                    (19,0,0))  # noqa
                time_splits = [
                    dt.time(
                        hour=hour, minute=minute, second=second
                    ) for (hour, minute, second) in time_splits_dict]
                # get the first energy state sampled after each time_split
                #   check if the dataframe goes to each time_split
                time_split_energies = []
                for time_split in time_splits:
                    if final_time < time_split:
                        time_split_energies.append(final_energy)
                    elif initial_time > time_split:
                        time_split_energies.append(initial_energy)
                    else:
                        time_split_energy = ev_day_df[
                            'vehicle_actualBatteryCapacity'][
                                ev_day_df.timestep_time >= dt.datetime(
                                    date.year, date.month, date.day,
                                    time_split.hour, time_split.minute,
                                    time_split.second)
                            ].iloc[0]/1000
                        time_split_energies.append(time_split_energy)
                # get difference of energy levels between each time-split
                energy_diffs = []
                for idx, time_split_energy in enumerate(time_split_energies):
                    if idx == 0:
                        prev_energy = initial_energy
                    energy_diff = time_split_energy - prev_energy
                    energy_diffs.append(energy_diff)
                    prev_energy = time_split_energy
                #   last energy diff:
                final_energy_diff = final_energy - time_split_energies[-1]

                # Create dictionary for the date
                keys = []
                vals = []
                keys.append("00:00:00 -> 23:59:59")
                vals.append(final_energy - initial_energy)
                prev_time = "00:00:00"
                for time_split, energy_diff in zip(time_splits, energy_diffs):
                    keys.append(f"{prev_time} -> {time_split}")
                    vals.append(energy_diff)
                    prev_time = time_split
                keys.append(f"{time_splits[-1]} -> 23:59:59")
                vals.append(final_energy_diff)

                stats_tree["stats"]["dates"].append(
                    {
                        "date": str(date),
                        "energy diffs": dict(zip(keys, vals))
                    }
                )

            # Create means and standard deviations of energy diffs
            sample_energy_dict = stats_tree[
                'stats']['dates'][0]['energy diffs']
            stats_tree['stats']['averages'] = {
                'energy diffs': dict.fromkeys(sample_energy_dict)
            }
            for period in sample_energy_dict.keys():
                stats_tree['stats']['averages']['energy ' +
                                                'diffs'][period] = {
                    'mean': None,
                    'stdev': None
                }
                energy_period = [date['energy diffs'][period]
                                 for date in stats_tree['stats']['dates']]
                try:
                    stats_tree['stats']['averages']['energy diffs'] \
                        [period]['mean'] = statistics.mean(energy_period)  # noqa
                    stats_tree['stats']['averages']['energy diffs'] \
                        [period]['stdev'] = statistics.stdev(energy_period)  # noqa
                except statistics.StatisticsError as e:
                    print(e)
                    input("Press any key to continue")

            # Write stat_file
            stats_path = vehicle_dir.joinpath('Outputs',
                                              f"stats_{vehicle_dir.name}.json")
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, 'w') as stats_file:
                json.dump(stats_tree, stats_file, indent=4)

            # Remove dates from the timestep_time column
            for ev_day_df in ev_dfs:
                ev_day_df['timestep_time'] = pd.to_datetime(
                    ev_day_df['timestep_time'].dt.time,
                    format='%H:%M:%S'
                )

            # TODO merge these changes into fleet_stats.
            # Create a mean dataframe
            #   Step 1: If any of the dataframes start later than the others,
            #   fill the missing energy values with the starting value.

            #       Get the lowest time in ev_dfs.
            #       TODO Maybe hardcode this to 00h00 -> 24h00
            earliest_time = min([ev_df['timestep_time'].min()
                                 for ev_df in ev_dfs])
            latest_time = max([ev_df['timestep_time'].max()
                               for ev_df in ev_dfs])

            #       Prepend empty rows to the ev_dfs so that they have the same
            #       lowest time.
            #           Use concat and group-by.
            #           Or use backfill.
            #           Add first row.
            #           Or use reindex function.
            for idx, ev_df in enumerate(ev_dfs):
                ev_df.set_index('timestep_time', inplace=True)

                ev_df = ev_df.reindex(pd.date_range(earliest_time, latest_time,
                                                    freq='s'))
                # fill nan values
                ev_df['vehicle_acceleration'] = \
                    ev_df['vehicle_acceleration'].fillna(0)
                ev_df['vehicle_energyCharged'] = \
                    ev_df['vehicle_energyCharged'].fillna(0)
                ev_df['vehicle_energyChargedInTransit'] = \
                    ev_df['vehicle_energyChargedInTransit'].fillna(0)
                ev_df['vehicle_energyChargedStopped'] = \
                    ev_df['vehicle_energyChargedStopped'].fillna(0)
                ev_df['vehicle_energyConsumed'] = \
                    ev_df['vehicle_energyConsumed'].fillna(0)
                ev_df['vehicle_speed'] = \
                    ev_df['vehicle_speed'].fillna(0)
                ev_df = ev_df.ffill().bfill()
                ev_df = ev_df.rename_axis('timestep_time').reset_index()
                ev_dfs[idx] = ev_df

            #       Modify the prepended rows so that their energy usage is
            #       equal to their first energy level.

            #   Create the mean date frame
            ev_df_mean = pd.concat(ev_dfs).groupby(['timestep_time']).mean()
            ev_df_mean.reset_index(level='timestep_time', inplace=True)

            # Create a mean plot
            plt_fig = self.__plot_summary_graph(
                ev_df_mean, plt_title="Simulation Output of experiment: "
                + vehicle_dir.name + " > Mean Plot")

            # Save the dataframe
            csv_path = vehicle_dir.joinpath('Outputs',
                                            f'{vehicle_dir.name}.csv')
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            ev_df_mean.to_csv(csv_path, index=False)

            # Save the plot
            if save_plots:
                # Create Graphs folder and save plot
                graphsdir = vehicle_dir.joinpath('Outputs', 'Graphs')
                graphsdir.mkdir(parents=True, exist_ok=True)
                save_path = graphsdir.joinpath('summary.svg')
                plt_fig.savefig(save_path)

        plt.show()
        input("Press Enter to close plots and " +
              "continue program execution...")
        plt.close('all')

    #def __gen_ev_box_plots(self) -> plt.Figure:
        # Read statistics from json files
        #

    def save_fleet_stats(self) -> None:
        # FIXME Make this function use
        # sub-functions of `save_ev_stats`
        print("\nGenerating fleet stats...")
        save_plots = input("Would you like to save the plots? [y]/n \n\t")
        save_plots = False if save_plots.lower() == 'n' else True

        def gen_ev_dfs_of_fleet() -> typ.Tuple[str,
                                               typ.Iterator[pd.DataFrame]]:
            """Generate dataframes of each taxi's mean data."""
            for ev_dir in self.__indiv_vehicle_dirs:
                # Obtain the EV's mean dataframe
                ev_df = self.__ev_csv_to_df(
                    ev_dir.joinpath('Outputs', f'{ev_dir.name}.csv'),
                    secs_to_dts=False, delim=',')
                yield (ev_dir.name, ev_df)

        # Initialise a statistics tree
        stats_tree: typ.Dict = {
            "stats": {
                "averages": None,
                "EVs": []
            }
        }

        # Get dataframes of each ev
        ev_names_and_dfs = [*gen_ev_dfs_of_fleet()]
        ev_names = [ev_name for (ev_name, _) in ev_names_and_dfs]
        ev_dfs = [ev_df for (_, ev_df) in ev_names_and_dfs]

        # Populate the statistics for the various EVs
        for ev_name, ev_df in ev_names_and_dfs:
            date = ev_df['timestep_time'][0].date()
            initial_time = ev_df['timestep_time'].iloc[0].time()
            final_time = ev_df['timestep_time'].iloc[-1].time()
            initial_energy = ev_df[
                'vehicle_actualBatteryCapacity'].iloc[0]/1000
            final_energy = ev_df[
                'vehicle_actualBatteryCapacity'].iloc[-1]/1000
            # `time_splits` which allow for arbitrary time-splitting
            # of statistics.
            time_splits_dict = ((6,0,0), (9,0,0), (12,0,0), (16,0,0),  # noqa
                                (19,0,0))  # noqa
            time_splits = [
                dt.time(
                    hour=hour, minute=minute, second=second
                ) for (hour, minute, second) in time_splits_dict]
            # get the first energy state sampled after each time_split
            #   check if the dataframe goes to each time_split
            time_split_energies = []
            for time_split in time_splits:
                if final_time < time_split:
                    time_split_energies.append(final_energy)
                elif initial_time > time_split:
                    time_split_energies.append(initial_energy)
                else:
                    time_split_energy = ev_df[
                        'vehicle_actualBatteryCapacity'][
                            ev_df.timestep_time >= dt.datetime(
                                date.year, date.month, date.day,
                                time_split.hour, time_split.minute,
                                time_split.second)
                        ].iloc[0]/1000
                    time_split_energies.append(time_split_energy)
            # get difference of energy levels between each time-split
            energy_diffs = []
            for idx, time_split_energy in enumerate(time_split_energies):
                if idx == 0:
                    prev_energy = initial_energy
                energy_diff = time_split_energy - prev_energy
                energy_diffs.append(energy_diff)
                prev_energy = time_split_energy
            #   last energy diff:
            final_energy_diff = final_energy - time_split_energies[-1]

            # Create dictionary for the date
            keys = []
            vals = []
            keys.append("00:00:00 -> 23:59:59")
            vals.append(final_energy - initial_energy)
            prev_time = "00:00:00"
            for time_split, energy_diff in zip(time_splits, energy_diffs):
                keys.append(f"{prev_time} -> {time_split}")
                vals.append(energy_diff)
                prev_time = time_split
            keys.append(f"{time_splits[-1]} -> 23:59:59")
            vals.append(final_energy_diff)

            stats_tree["stats"]["EVs"].append(
                {
                    "name": ev_name,
                    "energy diffs": dict(zip(keys, vals))
                }
            )


        # Create means and standard deviations of energy diffs
        sample_energy_dict = stats_tree[
            'stats']['EVs'][0]['energy diffs']
        stats_tree['stats']['averages'] = {
            'energy diffs': dict.fromkeys(sample_energy_dict)
        }
        for period in sample_energy_dict.keys():
            stats_tree['stats']['averages']['energy ' +
                                            'diffs'][period] = {
                'mean': None,
                'stdev': None
            }
            energy_period = [ev_stats['energy diffs'][period]
                             for ev_stats in stats_tree['stats']['EVs']]
            try:
                stats_tree['stats']['averages']['energy diffs'] \
                    [period]['mean'] = statistics.mean(energy_period)  # noqa
                stats_tree['stats']['averages']['energy diffs'] \
                    [period]['stdev'] = statistics.stdev(energy_period)  # noqa
            except statistics.StatisticsError as e:
                print(e)
                input("Press any key to continue")

        # Write stat_file
        stats_path = self.__agg_vehicle_dir.joinpath('Outputs',
                                                     'stats_fleet.json')
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as stats_file:
            json.dump(stats_tree, stats_file, indent=4)

        # Make *Mean* plot ####################################################

        earliest_time = min([ev_df['timestep_time'].min() for ev_df in ev_dfs])
        latest_time = max([ev_df['timestep_time'].max() for ev_df in ev_dfs])
        for idx, ev_df in enumerate(ev_dfs):
            ev_df.set_index('timestep_time', inplace=True)
            ev_df = ev_df.reindex(pd.date_range(earliest_time, latest_time,
                                                freq='s'))
            # fill nan values
            ev_df['vehicle_acceleration'] = \
                ev_df['vehicle_acceleration'].fillna(0)
            ev_df['vehicle_energyCharged'] = \
                ev_df['vehicle_energyCharged'].fillna(0)
            ev_df['vehicle_energyChargedInTransit'] = \
                ev_df['vehicle_energyChargedInTransit'].fillna(0)
            ev_df['vehicle_energyChargedStopped'] = \
                ev_df['vehicle_energyChargedStopped'].fillna(0)
            ev_df['vehicle_energyConsumed'] = \
                ev_df['vehicle_energyConsumed'].fillna(0)
            ev_df['vehicle_speed'] = \
                ev_df['vehicle_speed'].fillna(0)
            ev_df = ev_df.ffill().bfill()
            ev_df = ev_df.rename_axis('timestep_time').reset_index()
            ev_dfs[idx] = ev_df

        # Create a mean dataframe
        ev_df_mean = pd.concat(ev_dfs).groupby(['timestep_time']).mean()
        ev_df_mean.reset_index(level='timestep_time', inplace=True)

        # Create a summary mean plot of the fleet
        plt_figs = []
        plt_figs.append(self.__plot_summary_graph(
            ev_df_mean, plt_title="Simulation Output of experiment: Fleet > " +
            "Mean Plot"))

        # Create a mean plot of the fleet with ONLY Power
        plt_figs.append(self.__plot_multiple_lines(
            [ev_df_mean], plt_title="Simulation Output of experiment: Fleet " +
            "> Mean Plot", plt_type='Power'))

        # Make multiple-line power plot (mean plot of each EV in the fleet)
        plt_figs.append(self.__plot_multiple_lines(
            ev_dfs, 'Simulation output of experiment: Fleet > EV Means',
            'Power', ev_names))

        # Make multiple-line speed plot (mean plot of each EV in the fleet)
        plt_figs.append(self.__plot_multiple_lines(
            ev_dfs, 'Simulation output of experiment: Fleet > EV Means',
            'Speed', ev_names))

        # Make multiple-line distance plot (mean plot of each EV in the fleet)
        plt_figs.append(self.__plot_multiple_lines(
            ev_dfs, 'Simulation output of experiment: Fleet > EV Means',
            'Distance', ev_names))

        # Save the dataframe
        csv_path = self.__agg_vehicle_dir.joinpath('Outputs', "fleet.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        ev_df_mean.to_csv(csv_path, index=False)

        # Save the plots
        if save_plots:
            # Create Graphs folder and save plot
            graphsdir = self.__agg_vehicle_dir.joinpath('Outputs', 'Graphs')
            graphsdir.mkdir(parents=True, exist_ok=True)
            for idx, plt_fig in enumerate(plt_figs):
                save_path = graphsdir.joinpath(str(idx) + '.svg')
                plt_fig.savefig(save_path)

        plt.show()
        input("Press Enter to close plots and " +
              "continue program execution...")
        plt.close('all')


# %% Main #####################################################################
def run_results_analysis(scenario_dir: Path, **kwargs):
    ev_sim_dirs = [*scenario_dir.joinpath(
        'SUMO_Simulation', 'Simulation_Outputs').glob('*/*/')]

    data_analysis = Data_Analysis(scenario_dir, ev_sim_dirs)
    data_analysis.make_plots()
    data_analysis.save_ev_stats()
    data_analysis.save_fleet_stats()
