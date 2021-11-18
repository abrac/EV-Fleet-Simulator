#! /usr/bin/env python3

# TODO:
# - [ ] Refactor "indiv_ev" and "indiv_vehicle" to ev.
# - [ ] Refactor "agg_ev" and "agg_vehicle" to fleet.
# - [ ] Save pickled versions of plots. *
# XXX:
# - [ ] Remove breakpoints scattered everywhere.

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
from multiprocessing import Pool
import pickle
import gc
import statsmodels.api as sm
import data_processing_ev as dpr
from matplotlib.ticker import FuncFormatter

if "SUMO_HOME" in os.environ:
    xml2csv = Path(os.environ["SUMO_HOME"], "tools", "xml", "xml2csv.py")
else:
    sys.exit("Please declare environmental variable 'SUMO_HOME'.")

# plt.ion()
plt.style.use('default')

MY_DPI = 96
mm = 1 / 25.4  # millimeters in inches


def _y_fmt(y, pos):
    decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9]
    suffix  = ['G', 'M', 'k', '', 'm', 'u', 'n']
    if y == 0:
        return str(0)
    for i, d in enumerate(decades):
        if np.abs(y) >= d:
            val = y / float(d)
            signf = len(str(val).split('.')[1])
            if signf == 0:
                return '{val:d} {suffix}'.format(val=int(val), suffix=suffix[i])
            else:
                if signf == 1:
                    # print(val, signf)
                    if str(val).split(".")[1] == "0":
                        return '{val:d} {suffix}'.format(val=int(round(val)), suffix=suffix[i])
                tx = "{" + "val:.{signf}f".format(signf=signf) + "} {suffix}"
                return tx.format(val=val, suffix=suffix[i])
                # return y
    return y


# %% Analysis Class ###########################################################
class Data_Analysis:
    def __create_csvs(self, ev_sim_dirs: typ.Sequence[Path]) -> None:
        """Convert all battery.out.xml files to csv files and save them

        They will be saved in a corresponding folder in
        {Scenario_Dir}/Results/
        """

        # TODO Implement auto_run mode.
        battery_csvs = [*self.__scenario_dir.joinpath(
                        'Results').glob('*/*/Battery.out.csv')]
        if len(battery_csvs) == 0:
            _ = input("Would you like to convert all " +
                      "battery.out.xml files to csv? [y]/n \n\t")
            convert = (True if _.lower() != 'n' else False)
        else:
            _ = input("Would you like to re-convert all " +
                      "battery.out.xml files to csv? y/[n] \n\t")
            convert = (False if _.lower() != 'y' else True)

        if convert:
            _ = input("Would you like to skip existing csv files? y/[n] \n\t")
            skipping = (True if _.lower() == 'y' else False)

            print("\nConverting xml files to csv...")
            for ev_sim_dir in tqdm(ev_sim_dirs):
                # Try create ev_csv if it doesn't exist
                ev_name = ev_sim_dir.parents[0].name
                date = ev_sim_dir.name
                battery_csv = self.__scenario_dir.joinpath(
                    'Results', ev_name, date, 'Battery.out.csv')
                battery_xml = ev_sim_dir.joinpath("Battery.out.xml")
                if not battery_xml.exists():
                    continue
                if skipping and battery_csv.exists():
                    continue
                battery_csv.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(['python', xml2csv, '-o', battery_csv,
                                battery_xml])
                # Warn if battery_csv *still* doesn't exist
                if not battery_csv.exists():
                    logging.warning("Failed to create ev_csv in \n\t" +
                                    str(battery_csv.parent))

    def __init__(self, scenario_dir: Path,
                 ev_sim_dirs: typ.Sequence[Path], **kwargs) -> None:
        """
        Create a data_analysis class.
        """
        self.__scenario_dir = scenario_dir
        self.kwargs = kwargs
        self.input_data_fmt = kwargs.get('input_data_fmt',
                                         dpr.DATA_FMTS['GPS'])

        # Convert all battery.out.xml files to csv files
        # with Pool() as p:
        #    p.map(self.__create_csvs, ev_sim_dirs)

        self.__create_csvs(ev_sim_dirs)

        self.__battery_csv_paths = sorted(
            [*scenario_dir.joinpath('Results').glob('*/*/Battery.out.csv')]
        )
        self.__agg_vehicle_dir = scenario_dir.joinpath('SUMO_Simulation',
                                                       'Simulation_Outputs')
        self.__indiv_vehicle_dirs = sorted(
            [x for x in self.__agg_vehicle_dir.glob('*/') if x.is_dir()]
        )

    def __ev_csv_to_df(self, ev_csv: Path, warn_nan: bool = False,
                       secs_to_dts: bool = False, delim=';') -> pd.DataFrame:
        """
        Inputs:
            secs_to_dts: True if function must convert seconds to
                         datetime objects (in the case of csv input format) or
                         timedelta objects (in the case of GTFS input format)
                         in the 'timestep_time' column. It will be converted to
                         datetime if the input data format is GPS. Else, if the
                         input data format is GTFS, the colum will be converted
                         to timedeltas.
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
            if self.input_data_fmt == dpr.DATA_FMTS['GPS']:
                date_arr_str = ev_csv.parents[0].name.split('-')
                date_arr = [int(date_el) for date_el in date_arr_str]
                date = dt.date(*date_arr)
                ev_df['timestep_time'] = pd.to_datetime(
                    ev_df['timestep_time'], unit='s', origin=date)
            elif self.input_data_fmt == dpr.DATA_FMTS['GTFS']:
                ev_df['timestep_time'] = pd.to_timedelta(
                    ev_df['timestep_time'], unit='seconds')
            else:
                raise ValueError(dpr.DATA_FMT_ERROR_MSG)
        else:
            # This option is used if the column is already in a datetime
            # format. This is the case when generating fleet statistics.
            ev_df['timestep_time'] = pd.to_datetime(ev_df['timestep_time'])
        return ev_df

    def __plot_multiple_lines(self, ev_dfs: typ.Iterable[pd.DataFrame],
                              plt_title: str, plt_type: str = 'Power',
                              ev_names: typ.List[str] = None,
                              dpi: int = MY_DPI) -> plt.Figure:
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
        fig = plt.figure(figsize=(210/2*mm, 297/3*mm), dpi=dpi)

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
                'Power': lambda: ev_df['vehicle_energyConsumed'] * 3.6,
                'Speed': lambda: ev_df['vehicle_speed'] * 3.6,
                'Distance': lambda: pd.Series(dist / 1000)}
            plt_df = plt_df_func[plt_type]()

            time = (ev_df['timestep_time'][1:] if plt_type == 'Distance' else
                    ev_df['timestep_time'])

            if plt_type in ('Power', 'Speed'):
                plt_df = plt_df.rolling(rolling_window, center=True).mean()

            # Plot
            ax.plot(time, plt_df)

        if ev_names:
            ax.legend(ev_names)
        ax.axhline(color="0", lw=0.8)

        fig.tight_layout()

        return fig

    def __fill_area_between(self, ax, ev_dfs: typ.Iterable[pd.DataFrame],
                            ev_names: typ.List[str], color, MY_DPI: int = 96):
        """
        Plot graphs of multiple battery output data for a given simulation and
        fill area between them. Plot area on the axis given by `ax`.
        """
        for ev_df in ev_dfs:
            ev_df['timestep_time'] = pd.to_datetime(
                ev_df['timestep_time'].dt.time,
                format='%H:%M:%S'
            )

        # TODO: Make `rolling_window` function argument
        rolling_window = 3600  # seconds

        power_dfs = []
        dist_dfs = []
        time_dfs = []
        for ev_df in ev_dfs:
            power_df = ev_df['vehicle_energyConsumed'] * 3.6

            dist = integrate.cumtrapz(ev_df['vehicle_speed'], dx=1)
            dist_df = pd.Series(dist / 1000)
            dist_dfs.append(dist_df)

            time_df = ev_df['timestep_time']
            time_dfs.append(time_df)

            power_df = power_df.rolling(rolling_window, center=True).mean()
            power_dfs.append(power_df)
        power_min_df = np.nanmin(power_dfs, axis=0)
        power_max_df = np.nanmax(power_dfs, axis=0)
        # Plot
        # time = np.linspace(
        #     dt.time(hour=0), dt.time(hour=23, minute=59, second=59),
        #     len(power_max_df)
        # )
        min_time = np.nanmin(time_dfs)
        max_time = np.nanmax(time_dfs)
        time = pd.date_range(min_time, max_time, freq='S')
        # ax.plot(time, power_min_df, color=color, alpha=0.6)
        # ax.plot(time, power_max_df, color=color, alpha=0.6)
        ax.fill_between(time, power_min_df, power_max_df, color=color,
                        facecolor='#00000000', label="Power Distribution",
                        hatch='.....')
        ax.legend()
        ax.set_ylim(bottom=0, top=35)
        # ax.legend(ncol=2)

    def __plot_summary_graph(self, ev_df: pd.DataFrame,
                             plt_fig: plt.Figure = None,
                             plt_title: str = "Simulation Output",
                             dpi: int = MY_DPI
                             ) -> plt.Figure:
        """
        Plot the graph of battery output data for a given simulation.

        Inputs:
            ev_df pd.DataFrame:
                The DataFrame has a SUMO EV battery output. The battery output
                is in XML format, this must be converted to a CSV and then a
                Data-Frame.
            plt_fig plt.Figure:
                The figure on which to draw the plots. This is optional. If it
                is not provided, a new figure will be created. The title of the
                new figure and the dpi can be defined through optional
                arguments, `plt_title` and `MY_DPI`.
        """

        # TODO Separate each subplot into a new function.
        if plt_fig is None:
            new_fig = True
            plt_fig = plt.figure(figsize=(210 * mm, 297 * mm), dpi=dpi)
            plt.suptitle(plt_title)
            (ax_PvT, ax_VvT), (ax_EvT, ax_EvX), (ax_PvX, ax_VvX) = (
                plt_fig.subplots(3, 2))
        else:
            new_fig = False
            ax_PvT, ax_VvT, ax_EvT, ax_EvX, ax_PvX, ax_VvX = plt_fig.get_axes()

        time = ev_df['timestep_time']
        # Convert the timestep_time column to datetimes if the current dtype is
        # timedelta.
        if pd.api.types.is_timedelta64_ns_dtype(time):
            time = dt.datetime(2000, 1, 1, 0, 0) + time

        # Variables needed in plots
        # Approx_distance in km. Note: `ev_df['vehicle_speed']` is in m/s.
        dist = integrate.cumtrapz(ev_df['vehicle_speed'], dx=1) / 1000
        # TODO: Make `rolling_window` function argument
        rolling_window = 3600  # seconds

        power_df = ev_df['vehicle_energyConsumed'] * 3.6 * 1000  # 3.6 Wh/s in 1 kW
        power_df_rolling = ev_df['vehicle_energyConsumed'].rolling(
            rolling_window, center=True).mean() * 3.6 * 1000

        # ax_PvT.set_title('Power vs Time')
        ax_PvT.set_xlabel('Time')
        ax_PvT.set_ylabel('Power (W)')
        if new_fig:
            ax_PvT.plot(time, power_df, label="Instantaneous Power", lw=0.5,
                        c='0.5')
            ax_PvT.axhline(color="0", lw=0.8)
            plt.setp(ax_PvT.get_xticklabels(), rotation=45)
        if new_fig:
            ax_PvT.plot(time, power_df_rolling, label="Rolling Average Power",
                        c='0')
            ax_PvT.legend()
        else:
            ax_PvT.plot(time, power_df_rolling, label="Rolling Average Power")
        # mpl_align.yaxes(ax_PvT, 0, ax_P2vT, 0)
        ax_PvT.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter('%H:%M'))
        ax_PvT.yaxis.set_major_formatter(FuncFormatter(_y_fmt))

        if new_fig:
            # ax_PvX.set_title('Power vs Distance')
            ax_PvX.axhline(color="0", lw=0.8)
            ax_PvX.set_xlabel('Distance (km)')
            # Plot instantaneous power
            ax_PvX.set_ylabel('Instantaneous Power (W)')
            ax_PvX.plot(dist, power_df[1:])
            ax_PvX.axhline(color="0", lw=0.8)
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

        # plt.title('Energy vs Time')
        ax_EvT.plot(time, (ev_df['vehicle_actualBatteryCapacity'].iloc[0] -
                           ev_df['vehicle_actualBatteryCapacity']))
        ax_EvT.set_ylabel('Energy (Wh)')
        ax_EvT.set_xlabel('Time')
        plt.setp(ax_EvT.get_xticklabels(), rotation=45)
        ax_EvT.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter('%H:%M'))
        ax_EvT.yaxis.set_major_formatter(FuncFormatter(_y_fmt))

        if new_fig:
            # plt.title('Energy vs Distance')
            ax_EvX.plot(dist,
                        (ev_df['vehicle_actualBatteryCapacity'].iloc[1] -
                         ev_df['vehicle_actualBatteryCapacity'][1:]))
            ax_EvX.set_ylabel('Energy (Wh)')
            ax_EvX.set_xlabel('Distance (km)')
            ax_EvX.yaxis.set_major_formatter(FuncFormatter(_y_fmt))

            # If wanting to plot a line of best fit of the energy vs dist profile:
            # def fit_line2(x, y):
            #    """Return slope, intercept of best fit line."""
            #    X = sm.add_constant(x)
            #    model = sm.OLS(y, X, missing='drop') # ignores entires where x or y is NaN
            #    fit = model.fit()
            #    return fit, fit.params[1], fit.params[0] # return stderr via fit.bse
            #
            # fit, m, b = fit_line2(x, y)
            # N = len(dist) # could be just 2 if you are only drawing a straight line...
            # points = np.linspace(x.min(), x.max(), N)
            # sm.tools.eval_measures.rmse([points, m*points+b], [x, y])

        if new_fig:
            # plt.title('Speed vs Time')
            # Note, vehicle_speed is m/s
            # ∴ x m/s = x / 1000 * 3600 km/h = x * 3.6 km/h = y km/h
            ax_VvT.plot(time, ev_df['vehicle_speed'] * 3.6, lw=0.5, c='0.2')
            # plt.ylabel(r'$ Speed\ (km \cdot h^{-1}) $')
            ax_VvT.set_ylabel('Speed (km/h)')
            ax_VvT.set_xlabel('Time')
            plt.setp(ax_VvT.get_xticklabels(), rotation=45)
            ax_VvT.xaxis.set_major_formatter(
                matplotlib.dates.DateFormatter('%H:%M'))
            ax_VvT.set_ylim(bottom=0, top=20)
            ax_VvT.yaxis.set_major_formatter(FuncFormatter(_y_fmt))

        if new_fig:
            # plt.title('Speed vs Distance')
            ax_VvX.plot(dist, ev_df['vehicle_speed'][1:] * 3.6)
            ax_VvX.set_ylabel(r'$ Speed\ (km \cdot h^{-1}) $')
            ax_VvX.set_xlabel('Distance (km)')
            ax_VvX.yaxis.set_major_formatter(FuncFormatter(_y_fmt))

        if new_fig:
            plt_fig.tight_layout()

        return plt_fig

    def make_plots(self) -> None:
        """
        Asks a few questions and generates plots accordingly. The plots
        generated are for each sumo simulation.

        In case the input data format is GTFS, plots will be generated for each
        *instance* of the trip, as defined in frequencies.txt.
        """

        if self.input_data_fmt == dpr.DATA_FMTS['GPS']:

            print(f"\nAbout to plot {len(self.__battery_csv_paths)} results...")
            _ = input("\nWould you like to continue? y/[n]\n\t")
            if _.lower() != 'y':
                return
            print("\nGenerating plots...")

            for ev_csv in tqdm(self.__battery_csv_paths):
                # Get parent directory
                ev_dir = ev_csv.parent
                trip_instance = self.__ev_csv_to_df(ev_csv, secs_to_dts=True)
                # Plot
                plt_fig = self.__plot_summary_graph(
                    trip_instance, plt_title="Simulation Output of experiment: "
                    + f"{ev_dir.parent.name} > " + f"{ev_dir.name}")
                # Create Graphs folder and save plot
                graphsdir = ev_dir.joinpath('Graphs')
                graphsdir.mkdir(parents=True, exist_ok=True)
                save_path = graphsdir.joinpath('summary.svg')
                plt_fig.savefig(save_path)
                # Save interactive figure
                output_file = graphsdir.joinpath(
                    'summary.fig.pickle'
                )
                pickle.dump(plt_fig, open(output_file, 'wb'))
                plt.close('all')

        elif self.input_data_fmt == dpr.DATA_FMTS['GTFS']:

            print(f"\nAbout to process {len(self.__battery_csv_paths)} results...")
            _ = input("\nWould you like to continue? [y]/n\n\t")
            if _.lower() == 'n':
                return

            _ = input("\n Would you like to plot the results? If not, I will " +
                      "just compute the values/statistics. y/[n]")
            plotting = False if _.lower() != 'y' else True
            print(f"\nGenerating statistics{' and plots' if plotting else ''}...")

            # Read frequencies.txt
            frequencies_file = self.__scenario_dir.joinpath(
                '_Inputs', 'Traces', 'Original', 'GTFS', 'frequencies.txt')
            if frequencies_file.exists():
                frequencies_defined = True
                frequencies_df = pd.read_csv(frequencies_file,
                                             dtype={'trip_id': str})
            else:
                frequencies_defined = False

            # Read stop_times.txt
            stop_times_df = pd.read_csv(self.__scenario_dir.joinpath(
                '_Inputs', 'Traces', 'Original', 'GTFS', 'stop_times.txt'),
                dtype={'trip_id': str})

            def _get_departure_delay(trip_id: str) -> dt.timedelta:
                """Calculates the time taken from the beginning of the trip,
                   until the vehicle leaves the first stop. Based on the raw
                   GTFS data."""
                stop_times = stop_times_df[stop_times_df['trip_id'] == trip_id]
                hour, minute, second = [int(x) for x in
                    stop_times.iloc[0]['departure_time'].split(':')]
                departure_timedelta = dt.timedelta(
                    hours=int(hour), minutes=int(minute), seconds=int(second))
                hour, minute, second = [int(x) for x in
                    stop_times.iloc[0]['arrival_time'].split(':')]
                arrival_timedelta = dt.timedelta(
                    hours=int(hour), minutes=int(minute), seconds=int(second))
                departure_delay = departure_timedelta - arrival_timedelta
                return departure_delay

            for trip_csv in tqdm(self.__battery_csv_paths):
                trip_id = trip_csv.parents[1].name

                # Find the times that the trip applies.
                trip_df = self.__ev_csv_to_df(trip_csv, secs_to_dts=True)

                if frequencies_defined:
                    frequency_definitions = frequencies_df[
                        frequencies_df['trip_id'] == trip_id]
                    times = []
                    for _, frequency_definition in frequency_definitions.iterrows():
                        service_start_time = dt.datetime(
                            2000, 1, 1, *[int(x) for x in frequency_definition[
                                'start_time'].split(':')])
                        service_end_time = dt.datetime(
                            2000, 1, 1, *[int(x) for x in frequency_definition[
                                'end_time'].split(':')])
                        headway_time = dt.timedelta(
                            seconds=int(frequency_definition['headway_secs']))
                        num_trips = int(
                            (service_end_time - service_start_time).total_seconds() /
                            headway_time.total_seconds()) + 1
                        times.extend([service_start_time + headway_time * k for
                                      k in range(num_trips)])
                else:
                    stop_times = stop_times_df[
                        stop_times_df['trip_id'] == trip_id]
                    hour, minute, second = [
                        int(x) for x in
                        stop_times.iloc[0]['arrival_time'].split(':')]
                    times = [dt.datetime(2000, 1, 1, hour, minute, second)]

                departure_delay = _get_departure_delay(trip_id)

                if plotting:
                    # Plot the trip itinary.
                    plt_fig = self.__plot_summary_graph(
                        trip_df, plt_title="Simulation output of experiment: "
                        + f"{trip_id} > {trip_csv.parent.name} -- itinary")
                    # Create Graphs folder and save plot
                    graphsdir = trip_csv.parent.joinpath('Graphs')
                    graphsdir.mkdir(parents=True, exist_ok=True)
                    save_path = graphsdir.joinpath('trip_itinary.svg')
                    plt_fig.savefig(save_path)
                    # Save interactive figure
                    # output_file = graphsdir.joinpath(
                    #     'trip_itinary.fig.pickle')
                    # pickle.dump(plt_fig, open(output_file, 'wb'))
                    plt.close('all')

                trip_instances = []
                first_time = True
                for time in times:
                    trip_instance = trip_df.copy()
                    trip_instance['timestep_time'] = (
                        trip_instance['timestep_time'] + time -
                        departure_delay)
                    trip_instances.append(trip_instance)

                    if plotting:
                        # Add the trip instance onto the figure.
                        if first_time:
                            plt_fig = self.__plot_summary_graph(
                                trip_instance)
                            first_time = False
                        else:
                            plt_fig = self.__plot_summary_graph(
                                trip_instance, plt_fig)

                if plotting:
                    # Save the trip instances plot.
                    save_path = graphsdir.joinpath('trip_instances.svg')
                    plt_fig.savefig(save_path)

                    # Save interactive figure
                    # output_file = graphsdir.joinpath('trip_instances.fig.pickle')
                    # pickle.dump(plt_fig, open(output_file, 'wb'))
                    plt.close('all')

                # Compute aggregated trip_instances
                # ---------------------------------

                #   FIXME This is code duplication with the code in
                # `save_ev_stats`.

                # Remove dates from the timestep_time column
                for trip_instance in trip_instances:
                    trip_instance['timestep_time'] = pd.to_datetime(
                        trip_instance['timestep_time'].dt.time,
                        format='%H:%M:%S'
                    )

                # Finding the earliest start time and the lastest end time.
                earliest_time = min([trip_instance['timestep_time'].min()
                                     for trip_instance in trip_instances])
                latest_time = max([trip_instance['timestep_time'].max()
                                   for trip_instance in trip_instances])

                # Prepend empty rows to the ev_dfs so that they have the same
                # lowest time.
                for idx, trip_instance in enumerate(trip_instances):
                    trip_instance.set_index('timestep_time', inplace=True)

                    trip_instance = trip_instance.reindex(
                        pd.date_range(earliest_time, latest_time, freq='s'))
                    # Fill nan values
                    # trip_instance['vehicle_acceleration'] = \
                    #     trip_instance['vehicle_acceleration'].fillna(0)
                    # trip_instance['vehicle_energyCharged'] = \
                    #     trip_instance['vehicle_energyCharged'].fillna(0)
                    # trip_instance['vehicle_energyChargedInTransit'] = \
                    #     trip_instance['vehicle_energyChargedInTransit'].fillna(0)
                    # trip_instance['vehicle_energyChargedStopped'] = \
                    #     trip_instance['vehicle_energyChargedStopped'].fillna(0)
                    trip_instance['vehicle_actualBatteryCapacity'] = \
                        trip_instance['vehicle_actualBatteryCapacity'].\
                        ffill().bfill().astype('int32')
                    trip_instance['vehicle_energyConsumed'] = \
                        trip_instance['vehicle_energyConsumed'].fillna(0).\
                        astype('int16')
                    trip_instance['vehicle_speed'] = \
                        trip_instance['vehicle_speed'].fillna(0).astype('int8')
                    # Modify the prepended rows so that their energy usage is
                    # equal to their first energy level.
                    trip_instance = trip_instance.ffill().bfill()
                    trip_instance = trip_instance.\
                        rename_axis('timestep_time').reset_index()
                    trip_instances[idx] = trip_instance

                # Create the mean data frame
                trip_mean_profile = pd.concat(trip_instances).groupby(
                    ['timestep_time']).mean()
                trip_mean_profile.reset_index(level='timestep_time',
                                              inplace=True)

                # Save aggregated trip_instances dataframe
                trip_mean_profile.to_csv(trip_csv.parent.joinpath(
                    'Battery.out.aggregated.csv'))  # XXX FIXME Make sure that
                    # `ev_fleet_stats` reads *this* csv file in the case of
                    # GTFS mode.

                if plotting:
                    # Plot aggregated trip_instances
                    plt_fig = self.__plot_summary_graph(
                        trip_mean_profile, plt_title="Simulation Output of " +
                        f"experiment: {trip_id} > {trip_csv.parent.name}" +
                        f"Mean plot of {len(times)} instances")

                    # Save plot
                    save_path = graphsdir.joinpath(
                        'trip_instances_aggregated.svg')
                    plt_fig.savefig(save_path)

                    # output_file = graphsdir.joinpath(
                    #     'trip_instances_aggregated.fig.pickle')
                    # pickle.dump(plt_fig, open(output_file, 'wb'))
                    plt.close('all')

                # TODO Read calendar.txt and find the days of the week that the
                # trip applies.

                # TODO Read calendar_dates.txt and find dates of the year for
                # which the trip behaviour has exceptions.

        else:
            raise ValueError(dpr.DATA_FMT_ERROR_MSG)

        input("\nPress Enter to continue program execution...")
        plt.close('all')

    def save_ev_stats(self) -> None:
        _ = input("Would you like to compute statistics for each EV? [y]/n  ")
        if _.lower() == 'n':
            return

        print("\nGenerating individual ev stats...")
        save_plots = input("Would you like to save the plots? [y]/n \n\t")
        save_plots = False if save_plots.lower() == 'n' else True

        # XXX TODO Ask if wanting to plot. If not, skip that step.

        for vehicle_dir in tqdm(self.__indiv_vehicle_dirs):
            ev_name = vehicle_dir.name
            output_vehicle_dir = self.__scenario_dir.joinpath(
                'Results', ev_name, 'Outputs')

            # Initialise a statistics tree
            stats_tree: typ.Dict = {
                "stats": {
                    "averages": None,
                    "dates": []
                }
            }

            def __get_ev_dfs_of_ev(vehicle_dir: Path
                                   ) -> typ.Iterator[pd.DataFrame]:
                """Generate dataframes of days that taxi has valid data."""
                ev_dfs = []
                for battery_csv in self.__battery_csv_paths:
                    # If the input data format is GPS, use battery.out.csv.
                    # However, if the format is GTFS, use
                    # battery.out.aggregated.csv. This file incorporates the
                    # various trip_instances caused by frequencies.txt.
                    if self.input_data_fmt == dpr.DATA_FMTS['GPS']:
                        pass
                    elif self.input_data_fmt == dpr.DATA_FMTS['GTFS']:
                        battery_csv = battery_csv.parent.joinpath(
                            'Battery.out.aggregated.csv')
                    else:
                        raise ValueError(dpr.DATA_FMT_ERROR_MSG)
                    if vehicle_dir.name == battery_csv.parents[1].name:
                        # Obtain a dataframe for day
                        if self.input_data_fmt == dpr.DATA_FMTS['GPS']:
                            ev_df = self.__ev_csv_to_df(
                                battery_csv, secs_to_dts=True).iloc[:-1]
                                # FIXME Why is final_energy NaN? Making it  # noqa
                                # 2nd last for now.                         # noqa
                        elif self.input_data_fmt == dpr.DATA_FMTS['GTFS']:
                            ev_df = self.__ev_csv_to_df(
                                battery_csv, secs_to_dts=False,
                                delim=',').iloc[:-1]
                        else:
                            raise ValueError(dpr.DATA_FMT_ERROR_MSG)
                        # If ev_df ends on a different date than it started,
                        # strip away excess rows.
                        if (ev_df['timestep_time'].iloc[0].date() !=
                                ev_df['timestep_time'].iloc[-1].date()):
                            print("Warning! ev_df consists of " +
                                  "more than one date. Deleting excess rows. " +
                                  f"\nCheck {str(battery_csv.absolute())}")
                                    # FIXME Use logging module.
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
                        # Only keep the following columns:
                        # 'timestep_time', 'vehicle_actualBatteryCapacity',
                        # 'vehicle_energyConsumed', 'vehicle_speed'
                        ev_df = ev_df[[
                            'timestep_time',
                            'vehicle_actualBatteryCapacity',
                            'vehicle_energyConsumed',
                            'vehicle_speed'
                        ]]
                        ev_dfs.append(ev_df)
                return ev_dfs

            # Get dataframes of each ev-day
            ev_dfs = __get_ev_dfs_of_ev(vehicle_dir)

            # Populate the statistics for the various dates
            for ev_day_df in ev_dfs:
                date = ev_day_df['timestep_time'][0].date()
                initial_time = ev_day_df['timestep_time'].iloc[0].time()
                final_time = ev_day_df['timestep_time'].iloc[-1].time()
                initial_energy = ev_day_df[
                    'vehicle_actualBatteryCapacity'].iloc[0] / 1000
                final_energy = ev_day_df[
                    'vehicle_actualBatteryCapacity'].iloc[-1] / 1000
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

                del time_split_energies

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
                    # If the data format is gtfs, most probably, there will
                    # only be one day defined per trip, therefore, errors are
                    # ignored in the GTFS case.
                    if self.input_data_fmt == dpr.DATA_FMTS['GPS']:
                        print(e)
                        input("Press any key to continue")
                    elif self.input_data_fmt == dpr.DATA_FMTS['GTFS']:
                        pass
                    else:
                        raise ValueError(dpr.DATA_FMT_ERROR_MSG)

            # Write stat_file
            stats_path = output_vehicle_dir.joinpath(
                f"stats_{vehicle_dir.name}.json"
            )
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
                # XXX I am commenting out ones that I don't need.
                # ev_df['vehicle_acceleration'] = \
                #    ev_df['vehicle_acceleration'].fillna(0)
                # ev_df['vehicle_energyCharged'] = \
                #    ev_df['vehicle_energyCharged'].fillna(0)
                # ev_df['vehicle_energyChargedInTransit'] = \
                #    ev_df['vehicle_energyChargedInTransit'].fillna(0)
                # ev_df['vehicle_energyChargedStopped'] = \
                #    ev_df['vehicle_energyChargedStopped'].fillna(0)
                ev_df['vehicle_actualBatteryCapacity'] = \
                    ev_df['vehicle_actualBatteryCapacity'].ffill().bfill().\
                    astype('int32')
                ev_df['vehicle_energyConsumed'] = \
                    ev_df['vehicle_energyConsumed'].fillna(0).astype('int16')
                ev_df['vehicle_speed'] = \
                    ev_df['vehicle_speed'].fillna(0).astype('int8')
                ev_df = ev_df.ffill().bfill()
                ev_df = ev_df.rename_axis('timestep_time').reset_index()
                ev_dfs[idx] = ev_df

            #       Modify the prepended rows so that their energy usage is
            #       equal to their first energy level.

            #   Create the mean data frame
            ev_df_mean = pd.concat(ev_dfs).groupby(['timestep_time']).mean()
            ev_df_mean.reset_index(level='timestep_time', inplace=True)

            # Create a mean plot
            plt_fig = self.__plot_summary_graph(
                ev_df_mean, plt_title="Simulation Output of experiment: "
                + vehicle_dir.name + " > Mean Plot")

            # Save the dataframe
            csv_path = output_vehicle_dir.joinpath(f'{vehicle_dir.name}.csv')
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            ev_df_mean.to_csv(csv_path, index=False)

            # Save the plot
            if save_plots:
                # Create Graphs folder and save plot
                graphsdir = output_vehicle_dir.joinpath('Graphs')
                graphsdir.mkdir(parents=True, exist_ok=True)
                save_path = graphsdir.joinpath('summary.svg')
                plt_fig.savefig(save_path)
            plt.close()

            del ev_dfs
            del ev_df_mean
            del ev_day_df

        input("Press Enter to continue program execution...")

    # def __gen_ev_box_plots(self) -> plt.Figure:
        # Read statistics from json files
        #

    def save_fleet_stats(self) -> None:
        # FIXME Make this function use
        # sub-functions of `save_ev_stats`

        _ = input("Would you like to compute statistics for the fleet? [y]/n  ")
        if _.lower() == 'n':
            return

        print("\nGenerating fleet stats...")
        save_plots = input("Would you like to save the plots? [y]/n \n\t")
        save_plots = False if save_plots.lower() == 'n' else True

        def gen_ev_dfs_of_fleet() \
                -> typ.Tuple[str, typ.Iterator[pd.DataFrame]]:
                # FIXME This should be generator, not Tuple.
            """Generate dataframes of each taxi's mean data."""
            vehicle_result_dirs = sorted(
                [*self.__scenario_dir.joinpath('Results').glob('*')]
            )
            for ev_dir in vehicle_result_dirs:
                # Obtain the EV's mean dataframe
                ev_csv = ev_dir.joinpath('Outputs', f'{ev_dir.name}.csv')
                if ev_csv.exists():
                    ev_df = self.__ev_csv_to_df(
                        ev_csv, secs_to_dts=False, delim=',')
                else:
                    continue
                yield (ev_dir.name, ev_df)

        # Initialise a statistics tree
        stats_tree: typ.Dict = {
            "stats": {
                "averages": None,
                "EVs": []
            }
        }

        # Get dataframes of each ev
        print('Loading dataframes of each of the EVs...\n')
        ev_names_and_dfs = [*gen_ev_dfs_of_fleet()]
        ev_names = [ev_name for (ev_name, _) in ev_names_and_dfs]
        ev_dfs = [ev_df for (_, ev_df) in ev_names_and_dfs]

        print("Calculating fleet statistics...\n")
        # Populate the statistics for the various EVs
        for ev_name, ev_df in ev_names_and_dfs:
            date = ev_df['timestep_time'][0].date()
            initial_time = ev_df['timestep_time'].iloc[0].time()
            final_time = ev_df['timestep_time'].iloc[-1].time()
            initial_energy = ev_df[
                'vehicle_actualBatteryCapacity'].iloc[0] / 1000
            final_energy = ev_df[
                'vehicle_actualBatteryCapacity'].iloc[-1] / 1000
            # `time_splits` which allow for arbitrary time-splitting
            # of statistics.
            time_splits_dict = ((6, 0, 0), (9, 0, 0), (12, 0, 0), (16, 0, 0),
                                (19, 0, 0))
            # time_splits_dict = ((1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0),
            #                     (5, 0, 0), (6, 0, 0), (7, 0, 0), (8, 0, 0),
            #                     (9, 0, 0), (10, 0, 0), (11, 0, 0), (12, 0, 0),
            #                     (13, 0, 0), (14, 0, 0), (15, 0, 0), (16, 0, 0),
            #                     (17, 0, 0), (18, 0, 0), (19, 0, 0), (20, 0, 0),
            #                     (21, 0, 0), (22, 0, 0), (23, 0, 0))
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
                        'vehicle_actualBatteryCapacity'
                    ][
                        ev_df.timestep_time >= dt.datetime(
                            date.year, date.month, date.day,
                            time_split.hour, time_split.minute,
                            time_split.second)
                    ].iloc[0] / 1000
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

        # ---------------------------------------------------------------------
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
                energy_period_tree = \
                    stats_tree['stats']['averages']['energy diffs'][period]
                energy_period_tree['mean'] = statistics.mean(energy_period)
                energy_period_tree['stdev'] = statistics.stdev(energy_period)
            except statistics.StatisticsError as e:
                print(e)
                input("Press any key to continue")

        # Write stat_file
        stats_path = self.__scenario_dir.joinpath('Results', 'Outputs',
                                                  'stats_fleet.json')
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as stats_file:
            json.dump(stats_tree, stats_file, indent=4)

        # Make *Mean* plot ####################################################

        # Check if fleet mean energy has been calculated already.
        fleet_output_dir = self.__scenario_dir.joinpath('Results', 'Outputs')
        fleet_mean_file = fleet_output_dir.joinpath("fleet.csv")
        fleet_output_dir.mkdir(parents=True, exist_ok=True)

        use_existing_fleet_file = False
        # If so, ask if loading it.
        if fleet_mean_file.exists():
            _ = input(f"{fleet_mean_file.name} found at: \n\t {fleet_mean_file} \n" +
                      "Use this file? [Y]/n  ")
            use_existing_fleet_file = True if _.lower() != 'n' else False

        if use_existing_fleet_file:
            ev_df_mean = pd.read_csv(fleet_mean_file)
            ev_df_mean['timestep_time'] = pd.to_datetime(
                ev_df_mean['timestep_time'])
        else:
            earliest_time = min([ev_df['timestep_time'].min() for ev_df in ev_dfs])
            latest_time = max([ev_df['timestep_time'].max() for ev_df in ev_dfs])
            print("Preparing data for plotting...")
            for idx, ev_df in tqdm(enumerate(ev_dfs), total=len(ev_dfs)):
                ev_df.set_index('timestep_time', inplace=True)
                ev_df = ev_df.reindex(pd.date_range(earliest_time, latest_time,
                                                    freq='s'))
                # fill nan values
                # XXX I am commenting out ones that I don't need.
                # ev_df['vehicle_acceleration'] = \
                #    ev_df['vehicle_acceleration'].fillna(0)
                # ev_df['vehicle_energyCharged'] = \
                #    ev_df['vehicle_energyCharged'].fillna(0)
                # ev_df['vehicle_energyChargedInTransit'] = \
                #    ev_df['vehicle_energyChargedInTransit'].fillna(0)
                # ev_df['vehicle_energyChargedStopped'] = \
                #    ev_df['vehicle_energyChargedStopped'].fillna(0)
                ev_df['vehicle_actualBatteryCapacity'] = \
                    ev_df['vehicle_actualBatteryCapacity'].ffill().bfill().\
                    astype('int32')
                ev_df['vehicle_energyConsumed'] = \
                    ev_df['vehicle_energyConsumed'].fillna(0).astype('int16')
                ev_df['vehicle_speed'] = \
                    ev_df['vehicle_speed'].fillna(0).astype('int8')
                ev_df = ev_df.ffill().bfill()
                ev_df = ev_df.rename_axis('timestep_time').reset_index()
                ev_dfs[idx] = ev_df

            # Create a mean dataframe
            # XXX: TODO: Find a more memory efficient way of doing this.
            # `pd.concat` seems to duplicate everything in memory...
            ev_dfs = pd.concat(ev_dfs)
            ev_df_mean = ev_dfs.groupby(['timestep_time']).mean()

            # Save the dataframe
            ev_df_mean.to_csv(fleet_mean_file, index=True)
            ev_df_mean.reset_index(level='timestep_time', inplace=True)

        # TODO Check if it is necessary to delete this.
        # del ev_dfs
        del ev_names_and_dfs

        if self.input_data_fmt == dpr.DATA_FMTS['GTFS']:
            num_vehicles = int(input("How many vehicles are in the study? " +
                                     "(Enter an integer)  "))
            # Calculate the total number of trip instances. Multiply ev_df_mean
            # by that number to get the *total* energy profile of the eMBT
            # system. Divide that profile by the number of taxis in the city to
            # get the average energy profile per taxi.
            frequencies_file = self.__scenario_dir.joinpath(
                '_Inputs', 'Traces', 'Original', 'GTFS', 'frequencies.txt')
            if frequencies_file.exists():
                frequencies_defined = True
                frequencies_df = pd.read_csv(frequencies_file,
                                             dtype={'trip_id': str})
            else:
                frequencies_defined = False

            print("Calculating the total number of trips in the system...")
            total_trip_instances = 0
            for trip_id in tqdm(ev_names):
                if frequencies_defined:
                    frequency_definitions = frequencies_df[
                        frequencies_df['trip_id'] == trip_id]
                    for _, frequency_definition in frequency_definitions.iterrows():
                        service_start_time = dt.datetime(
                            2000, 1, 1, *[int(x) for x in frequency_definition[
                                'start_time'].split(':')])
                        service_end_time = dt.datetime(
                            2000, 1, 1, *[int(x) for x in frequency_definition[
                                'end_time'].split(':')])
                        headway_time = dt.timedelta(
                            seconds=int(frequency_definition['headway_secs']))
                        num_trips = int(
                            (service_end_time - service_start_time).total_seconds() /
                            headway_time.total_seconds()) + 1
                        total_trip_instances += num_trips
                else:
                    total_trip_instances += 1
            input(f"Total trip instances: {total_trip_instances}.\n" +
                  "Press any key to continue...")

        # Create a summary mean plot of the fleet
        plt_figs = {}
        plt_figs['mean_plot'] = self.__plot_summary_graph(
            ev_df_mean, plt_title="Simulation Output of experiment: Fleet > " +
            "Mean Plot")

        # Create a mean plot of the fleet with ONLY Power
        plt_figs['power_plot'] = self.__plot_multiple_lines(
            [ev_df_mean], plt_title="Simulation Output of experiment: Fleet " +
            "> Mean Plot", plt_type='Power')

        if self.input_data_fmt == dpr.DATA_FMTS['GTFS']:

            ev_df_mean_tmp = ev_df_mean.copy()

            ev_df_mean_tmp[['vehicle_actualBatteryCapacity',
                            'vehicle_energyConsumed', 'vehicle_speed']] = \
                ev_df_mean_tmp[['vehicle_actualBatteryCapacity',
                                'vehicle_energyConsumed',
                                'vehicle_speed']] * total_trip_instances

            plt_figs['total_plot'] = self.__plot_summary_graph(
                ev_df_mean_tmp, plt_title=
                "Simulation Output of experiment: Fleet > Total Plot")

            ev_df_mean_tmp[['vehicle_actualBatteryCapacity',
                            'vehicle_energyConsumed', 'vehicle_speed']] = \
                ev_df_mean_tmp[['vehicle_actualBatteryCapacity',
                                'vehicle_energyConsumed',
                                'vehicle_speed']] / num_vehicles

            plt_figs['ev_mean_plot'] = self.__plot_summary_graph(
                ev_df_mean_tmp, plt_title=
                "Simulation Output of experiment: Fleet > EV Mean Plot")


        if self.input_data_fmt != dpr.DATA_FMTS['GTFS']:
            # color = plt_figs[0].get_axes()[0].get_lines()[2].get_color()
            #self.__fill_area_between(plt_figs['mean_plot'].get_axes()[0],
                                     #ev_dfs, ev_names, '0')

            # color = plt_figs[1].get_axes()[0].get_lines()[0].get_color()
            #self.__fill_area_between(plt_figs['power_plot'].get_axes()[0],
                                     #ev_dfs, ev_names, '0')

            # Make multiple-line power plot (mean plot of each EV in the fleet)
            plt_figs['power_per_ev'] = self.__plot_multiple_lines(
                ev_dfs, 'Simulation output of experiment: Fleet > EV Means',
                'Power', ev_names)

            # Make multiple-line speed plot (mean plot of each EV in the fleet)
            plt_figs['speed_per_ev'] = self.__plot_multiple_lines(
                ev_dfs, 'Simulation output of experiment: Fleet > EV Means',
                'Speed', ev_names)

            # Make multiple-line distance plot (mean plot of each EV in the fleet)
            plt_figs['distance_per_ev'] = self.__plot_multiple_lines(
                ev_dfs, 'Simulation output of experiment: Fleet > EV Means',
                'Distance', ev_names)

        # Save the plots
        if save_plots:
            # Create Graphs folder and save plot
            graphsdir = fleet_output_dir.joinpath('Graphs')
            graphsdir.mkdir(parents=True, exist_ok=True)
            for key, plt_fig in plt_figs.items():
                save_file = graphsdir.joinpath(key + '.svg')
                plt_fig.savefig(save_file)
                save_file = graphsdir.joinpath(key + '.pdf')
                plt_fig.savefig(save_file)
                # As pickle:
                fig_file = graphsdir.joinpath(key + '.fig.pickle')
                pickle.dump(plt_fig, open(fig_file, 'wb'))

        plt.show()
        input("Press Enter to close plots and " +
              "continue program execution...")
        plt.close('all')


# %% Main #####################################################################
def run_ev_results_analysis(scenario_dir: Path, **kwargs):

    ev_sim_dirs = sorted([*scenario_dir.joinpath(
        'SUMO_Simulation', 'Simulation_Outputs').glob('*/*/')])

    data_analysis = Data_Analysis(scenario_dir, ev_sim_dirs, **kwargs)
    # XXX GTFS: No changes will be required in `make_plots()`. The function
    # will plot each day of each trip once. The plots will start at midnight,
    # as currently defined in the simulation results. Additionally (and I will
    # do this only later), this function will have an "interactive mode", which
    # will allow you to view the plot of any *instance* of the trip.
    data_analysis.make_plots()
    # XXX GTFS: `save_ev_stats()` will do two things. Firstly, it will combine
    # days in multi-day trips (those that take over 24 hours to complete). This
    # is the default behaviour of this function for GPS inputs. Secondly, the
    # function will use frequencies.txt data to figure out how often the trip
    # is serviced. It will then create an energy usage profile which represents
    # the trip. E.g. If a 30 minute trip is serviced, from 7--8 am, at 20
    # minute intervals, the average energy usage will be higher when two taxis
    # are present (i.e. from 07:20--07:30, 07:40--07:50, and 08:00--08:10.
    # TODO: Check if GTFS specifies that a bus should leave at 8 am in a case
    # like this.) Additionally (and I will do this only later), this function
    # will have an interactive mode which will allow you to view the aggregated
    # plot of any calendar date of this trip.
    data_analysis.save_ev_stats()
    # XXX GTFS: Using the 2nd output of the above function,
    # `save_fleet_stats()` will compute the average pprofile of all the trips
    # together. Additionally (and I will do this only later), this function
    # will aggregagate the results from *all dates* to find th averae profile
    # of all the trips, across all the dates over a year.
    data_analysis.save_fleet_stats()


if __name__ == "__main__":
    scenario_dir = Path(os.path.abspath(__file__)).parents[1]
    run_ev_results_analysis(scenario_dir)
    scenario_dir = Path(os.path.abspath(__file__)).parents[1]
    run_ev_results_analysis(scenario_dir)
