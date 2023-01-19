#!/usr/bin/env python3

import os
import pandas as pd
from scipy import integrate, interpolate
from pathlib import Path
import typing as tp
from tqdm import tqdm
import datetime as dt
import time
import calendar
import matplotlib.pyplot as plt
import pickle
import logging
import numpy as np
import matplotlib as mpl
import json
import data_processing_ev as dpr


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        """ Special json encoder for numpy types """
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def _load_irradiance_data(scenario_dir: Path, year: int) -> tp.Callable:
    # Load irradiance data as dataframe
    irradiance_file = scenario_dir.joinpath('REG_Simulation', 'Results',
                                            'POA_Irradiance', 'Data',
                                            f'{year}.csv')
        # TODO Don't rely on just one year, maybe take the average across
        # multiple years... And maybe keep each month seperate.
    pv_output_df = pd.read_csv(irradiance_file)
    pv_output_df.columns = ['DateTime', 'P_Out']

    # Convert first column of hourly times to to datetime objects.
    pv_output_df['DateTime'] = (
        pd.Timestamp(year=year, month=1, day=1, hour=0, minute=0) +
        pd.to_timedelta(pv_output_df['DateTime'], unit='H'))
    pv_output_df = pv_output_df.set_index('DateTime')

    # Apply efficiency constant to irradiance output.
    eff = 0.80 * 0.20
    pv_output_df['P_Out'] *= eff  # Multiply irradiance by efficiency constant.

    # Create a function which interpolates between points of pv_output_df. This
    # will be used for integration.
    f_pv_out = interpolate.interp1d(
        # Make the x-values seconds since the epoch.
        x=pd.Series(pv_output_df.index).apply(
            lambda datetime: time.mktime(datetime.timetuple())),
        y=pv_output_df['P_Out'],
        fill_value='extrapolate')

    return f_pv_out


def run_pv_results_analysis(scenario_dir: Path, plot_blotches: bool = False,
                            figsize=(4, 3), **kwargs):
    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])
    # Load irradiance data
    # FIXME Don't hardcode this. Don't even ask the year.
    _ = dpr.auto_input("For which year is the solar irradiation data? "
                       "(Leave blank for 2017.) \n\t", '2017', **kwargs)
    try:
        year = int(_)
    except ValueError:
        year = 2017

    # Directories for saving CSVs and figures.
    csv_dir = scenario_dir.joinpath('REG_Simulation', 'csvs')
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = scenario_dir.joinpath('REG_Simulation', 'graphs')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load the PV irradiance function.
    f_pv_out = _load_irradiance_data(scenario_dir, year)

    # Load list of stop-arrivals and -durations as DataFrame.
    stops_file = scenario_dir.joinpath('Temporal_Clusters', 'Clustered_Data',
                                       'time_clusters_with_dates.csv')
    stops_df = pd.read_csv(stops_file).set_index(['EV_Name', 'Date'])

    # Check if pv_potentials has been calculated already.
    pv_potentials_file = csv_dir.joinpath('pv_potentials.csv')
    use_existing_pv_file = False
    # If so, ask if loading it.
    if pv_potentials_file.exists():
        _ = dpr.auto_input(
            f"pv_potentials.csv found at: \n\t {pv_potentials_file}"
            "\nUse this file? [y]/n  ", 'y', **kwargs)
        use_existing_pv_file = True if _.lower() != 'n' else False

    # If loading it, load it, else regenerate pv_potentials.
    if use_existing_pv_file:
        pv_potentials = pd.read_csv(pv_potentials_file)
    else:
        # Integrate all the energy charged at all stopped times.
        pv_potentials = []
        integration_errors = []
        long_stops_count = 0
        for _, stop in tqdm(stops_df.iterrows(), total=len(stops_df)):
            # Get the arrival and departure intervals of the stop-event.

            # Assume a data-error if the stop-duration is lonnger than 24 hrs.
            if stop['Stop_Duration'] >= 24:
                long_stops_count += 1
                continue

            # Get the date.
            # FIXME: Handle case of leap year (i.e. Feb 29 in date components)
            date = dt.date(year,
                           *[int(dc) for dc in stop.name[1].split('-')[1:]])

            # Get the beginning of the stop.
            stop_beg_float = stop['Stop_Arrival']
            # Seperate the stop time in to a tuple of (hours, minutes).
            stop_beg = (int(stop_beg_float), int((stop_beg_float % 1) * 60))
            # Convert it to a datetime.
            stop_beg = dt.datetime(date.year, date.month, date.day, *stop_beg)
            # Convert it to a float which represents the seconds since the
            # epoch.
            stop_beg = time.mktime(stop_beg.timetuple())

            # Get the end of the stop.
            stop_end_float = stop_beg_float + stop['Stop_Duration']
            # Seperate the stop time in to a tuple of (hours, minutes).
            stop_end = [int(stop_end_float), int((stop_end_float % 1) * 60)]
            # Handle case where stop extends to the next day:
            while stop_end[0] >= 24:
                stop_end[0] -= 24
                date += dt.timedelta(1)
            # Convert it to a datetime.
            stop_end = dt.datetime(date.year, date.month, date.day, *stop_end)
            # Convert it to a float which represents the seconds since the
            # epoch.
            stop_end = time.mktime(stop_end.timetuple())

            # Integrate to find the PV energy for this stop-event.
            # Result is in Ws/m^2
            stop_pv_potential, error = integrate.quad(f_pv_out, stop_beg,
                                                      stop_end)

            # Append the results of the integration to the result-lists.
            integration_errors.append(error)
            stop_copy = stop.copy()
            stop_copy['PV_Charge_Pot_Per_m2'] = stop_pv_potential
            pv_potentials.append(stop_copy)

        # Create a warning if there are discarded stops.
        if long_stops_count:
            logging.warn(f'Stops discarded: {long_stops_count}')
            print(f'Stops discarded: {long_stops_count}')
        else:
            print("No stops discarded.")
        # Convert the pv_potentials list into a dataframe.
        pv_potentials = pd.DataFrame(pv_potentials)
        pv_potentials.index = pd.MultiIndex.from_tuples(pv_potentials.index)
        pv_potentials.index.names = ['EV_Name', 'Date']
        pv_potentials = pv_potentials.reset_index()
        # Save the pv_potentials dataframe.
        pv_potentials_file.parent.mkdir(parents=True, exist_ok=True)
        pv_potentials.to_csv(pv_potentials_file, index=False)

    # Make date index-level consist of date-time objects.
    pv_potentials['Date'] = pd.to_datetime(pv_potentials.reset_index()['Date'])
    pv_potentials = pv_potentials.set_index(['EV_Name', 'Date'])

    # For each date, sum up the total charging potential for that date.
    pv_potentials_per_day = pv_potentials.\
        reset_index().groupby(['EV_Name', 'Date']).\
        sum()[['Stop_Duration', 'PV_Charge_Pot_Per_m2']]

    # For each taxi, fill missing dates with charging potentials of zero.
    # Read csv file with filtered dates.
    filtered_dates = pd.read_csv(
        scenario_dir.joinpath('Spatial_Clusters', 'Filtered_Traces',
                              'dirty__dates_remaining.csv')
    )
    # Convert the dates to date-time objects, and set it as the index.
    filtered_dates['Date'] = pd.to_datetime(filtered_dates['Date'])
    filtered_dates[['Stop_Duration', 'PV_Charge_Pot_Per_m2']] = 0
    filtered_dates = filtered_dates.set_index(['EV_Name', 'Date'])
    # Merge into `pv_potentials_per_day` the dates which are not in
    #   `pv_potentials_per_day`, but are in `filtered_dates`.
    pv_potentials_per_day = pd.concat([
        pv_potentials_per_day,
        filtered_dates[~filtered_dates.index.isin(pv_potentials_per_day.index)]
    ]).sort_index()

    # TODO Remove the below. We are no longer going to fill all missing dates.
    """
    ev_names = sorted(set(
        pv_potentials_per_day.index.get_level_values('EV_Name')
    ))
    # list of ev_dfs:
    ev_dfs = []
    for ev_name in ev_names:
        df = pv_potentials_per_day.loc[ev_name]
        min_date = min(df.index.get_level_values('Date'))
        max_date = max(df.index.get_level_values('Date'))
        days_in_max_month = calendar.monthrange(year, max_date.month)[1]

        fill_start = pd.to_datetime(
            dt.date(min_date.year, min_date.month, 1)
        )
        fill_end = pd.to_datetime(
            dt.date(max_date.year, max_date.month, days_in_max_month)
        )

        idx = pd.date_range(fill_start, fill_end)
        df = df.reindex(idx, fill_value=0)
        ev_dfs.append(df)

    # Join all the small dfs into one.
    pv_potentials_per_day = pd.concat(
        ev_dfs, keys=ev_names, names=['EV_Name', 'Date']
    )
    """

    # Get average daily pv potential over each month.
    date_index = pv_potentials_per_day.index.get_level_values('Date')
    ev_name_index = pv_potentials_per_day.index.get_level_values('EV_Name')
    _ = dpr.auto_input(
        "Would you like to obtain the monthly average by adding the "
        "values and dividing by dates on record, or by dividing by "
        "dates in month?  [record]/month  \n\t", 'record', **kwargs)
    divide_by_days_in_month = False if _.lower() != 'month' else True
    if divide_by_days_in_month:
        # Get the sum of the pv potentials for each month.
        pv_potentials_per_month = pv_potentials_per_day.groupby(
            [ev_name_index, date_index.year, date_index.month]
        ).sum()
        pv_potentials_per_month.index.names = ['EV_Name', 'Year', 'Month']
        # Initialise a list, in which to append Sieries's.
        pv_potentials_month_list = []
        for _, pv_potential_month in pv_potentials_per_month.iterrows():
            # Number of days in month.
            curr_year = pv_potential_month.name[1]
            curr_month = pv_potential_month.name[2]
            days_in_month = calendar.monthrange(curr_year, curr_month)[1]
            # Divide total energy by number of days in that month.
            average_potential = \
                pv_potential_month['PV_Charge_Pot_Per_m2'] / days_in_month
            average_stop = \
                pv_potential_month['Stop_Duration'] / days_in_month
            # Create a Series (row of DataFrame) with the value, and append to
            # list.
            row = pd.Series(
                {
                    'Stop_Duration': average_stop,
                    'PV_Charge_Pot_Per_m2': average_potential},
                name=pv_potential_month.name)
            pv_potentials_month_list.append(row)
        # Convert the list to a DataFrame.
        pv_potentials_month_avg = pd.DataFrame(
            pv_potentials_month_list,
            pd.MultiIndex.from_tuples(
                [row.name for row in pv_potentials_month_list],
                names=['EV_Name', 'Year', 'Month']
            )
        )
    else:
        pv_potentials_month_avg = pv_potentials_per_day.groupby(
            [ev_name_index, date_index.year, date_index.month]
        ).mean()
        pv_potentials_month_avg.index.names = ['EV_Name', 'Year', 'Month']

    # Get average energy of months across years:
    pv_potentials_avg_of_month_avgs_per_taxi = pv_potentials_month_avg.\
        reset_index().groupby(['EV_Name', 'Month']).\
        mean()[['Stop_Duration', 'PV_Charge_Pot_Per_m2']]  # TODO Bad variable name.

    # Get average energy of months across years and across taxis:
    pv_potentials_avg_of_month_avgs = pv_potentials_month_avg.\
        reset_index().groupby(['Month']).\
        mean()[['Stop_Duration', 'PV_Charge_Pot_Per_m2']]

    # Get approximate monthly total energy from PV:
    pv_potentials_total_per_month = pv_potentials_avg_of_month_avgs.copy()
    pv_potentials_total_per_month['PV_Charge_Pot_Per_m2'] *= 30

    # Get *total* pv energy generated by the solar panel
    # (not only energy used for charging EV):
    daily_pv_generated_p_month = []
    integration_errors = []
    for month in range(1, 13):
        days_in_month = calendar.monthrange(year, month)[1]
            # FIXME Warning: the year used here is different than the year
            # used to calculate the days_in_month for the
            # pv_charging_potential. Do a reverse search of "days_in_month" to
            # find the section that I am talking about.
        start = time.mktime(dt.datetime(year, month, 1, 0).timetuple())
        end = time.mktime(dt.datetime(year, month, days_in_month, 23, 59, 59
                                      ).timetuple())
        month_pv_generated, error = integrate.quad(f_pv_out, start, end)
        daily_pv_generated_p_month.append(month_pv_generated / days_in_month)
        integration_errors.append(error / days_in_month)
    daily_pv_generated_p_month = pd.DataFrame(
        daily_pv_generated_p_month, index=range(1, 13),
        columns=['Avg Daily PV Generation'])
    daily_pv_generated_p_month.index.names = ['Month']

    # Check if pv_potentials has been calculated already.
    pv_generated_file = csv_dir.joinpath('daily_pv_generated.csv')
    use_existing_pv_file = False
    # If so, ask if loading it.
    if pv_generated_file.exists():
        _ = dpr.auto_input("daily_pv_generated.csv found at: \n\t "
            f"{pv_generated_file} \n Use this file? [y]/n",
            'y', **kwargs)
        use_existing_pv_file = True if _.lower() != 'n' else False

    # If loading it, load it, else regenerate pv_potentials.
    if use_existing_pv_file:
        daily_pv_generated = pd.read_csv(pv_generated_file)
        daily_pv_generated = daily_pv_generated.set_index(['Date'])
    else:
        daily_pv_generated = []
        integration_errors = []
        days = 367 if calendar.isleap(year) else 366
        for day in tqdm(range(1, days)):
                # FIXME Warning: the year used here is different than the year
                # used to calculate the days_in_month for the
                # pv_charging_potential. Do a reverse search of "days_in_month" to
                # find the section that I am talking about.
            beginning_of_year = dt.datetime(year, 1, 1, 0)
            start_date = beginning_of_year + dt.timedelta(day - 1)
            start = time.mktime(start_date.timetuple())
            end = time.mktime(
                dt.datetime(start_date.year, start_date.month, start_date.day,
                            23, 59, 59).timetuple())
            day_pv_generated, error = integrate.quad(f_pv_out, start, end)
            day_pv_generated = pd.Series(
                day_pv_generated, index=['Daily_PV_Generation'],
                name=start_date.date())
            daily_pv_generated.append(day_pv_generated)
            integration_errors.append(error)

        daily_pv_generated = pd.DataFrame(daily_pv_generated,
                                          columns=['Daily_PV_Generation'])
        daily_pv_generated.index.names = ['Date']
        daily_pv_generated.to_csv(pv_generated_file)

    daily_pv_generated.index = pd.to_datetime(daily_pv_generated.index)

    # Save csv files of the dataframes for future reference.
    pv_potentials_per_day.to_csv(
        csv_dir.joinpath('pv_potentials_per_day.csv'))
    pv_potentials_month_avg.to_csv(
        csv_dir.joinpath('pv_potentials_month_avg.csv'))
    pv_potentials_avg_of_month_avgs.to_csv(
        csv_dir.joinpath('pv_potentials_avg_of_month_avgs.csv'))
    pv_potentials_avg_of_month_avgs_per_taxi.to_csv(
        csv_dir.joinpath('pv_potentials_avg_of_month_avgs_per_taxi.csv'))
    daily_pv_generated_p_month.to_csv(
        csv_dir.joinpath('daily_pv_generated_p_month.csv'))

    # Generate bar plots of the charging potential per m2.
    # Average energy charging potential for various months of the year.
    fig_bar_chart = plt.figure(figsize=figsize)
    width = 0.3
    plt.bar(pv_potentials_avg_of_month_avgs.index - width / 2,
            pv_potentials_avg_of_month_avgs['PV_Charge_Pot_Per_m2'] / 3600000,
            width=width)
    plt.bar(daily_pv_generated_p_month.index + width / 2,
            daily_pv_generated_p_month['Avg Daily PV Generation'] / 3600000,
            width=width)
    plt.ylabel('Average energy in a day (kWh/m2)')
    plt.xlabel('Month of year')
    plt.legend(['Energy charged directly from PV',
                'Total PV energy generated'])
    plt.tight_layout()

    # Plot box-plot version of the above figure.
    ev_names = sorted(set(
        pv_potentials_avg_of_month_avgs_per_taxi.index.\
        get_level_values('EV_Name')))
    for i, ev_name in enumerate(ev_names):
        # plt.subplot(3, 3, i+1)  # FIXME: Uncomment me!
        plt.figure(figsize=(3, 2))  # FIXME: Remove me!
        # Separate the `pv_potentials_avg_of_month_avgs` dataframe into a list
        # of 12 dataframes, each representing a month of the year.
        date_index = pv_potentials_per_day.loc[ev_name].index.get_level_values('Date')
        pv_potentials_all_months = [
            pv_potentials_per_day.loc[ev_name].set_index([date_index.month]).\
            loc[j] for j in set(date_index.month)]

        pv_potentials_all_months_ev = []
        for pv_potentials_ev in pv_potentials_all_months:
            if type(pv_potentials_ev) is pd.DataFrame:
                pv_potential = pv_potentials_ev['PV_Charge_Pot_Per_m2'] / 3600000
            else:
                pv_potentials_ev = pv_potentials_ev.to_frame().T
                pv_potential = pv_potentials_ev['PV_Charge_Pot_Per_m2'] / 3600000
            pv_potentials_all_months_ev.append(pv_potential)

        plt.boxplot(pv_potentials_all_months_ev,
                    medianprops={'color': 'black'},
                    positions=list(set(date_index.month)),
                    flierprops={'marker': '.'})
        box_stats = {
            'box stats': [
                mpl.cbook.boxplot_stats(pv_potentials_ev.values)[0] for
                pv_potentials_ev in pv_potentials_all_months_ev]}
        with open(csv_dir.joinpath(f'box_pv_potential_{ev_name}.json'),
                  'w') as f:
            json.dump(box_stats, f, cls=NumpyEncoder, indent=4)

        # plt.ylabel('Energy charged in a day (kWh/m2)')  # FIXME: Uncomment
        # plt.xlabel('Month of year')  # FIXME: Uncomment
        # plt.title(ev_name)  # FIXME: Uncomment
        # plt.xticks(range(1,13), range(1,13))
        plt.ylim((-0.06, 1.3))
        # plt.xlim((0.5, 12.5))
        plt.tight_layout()
        if (plot_blotches):
            # Plot the stop_events which make up the box-plots.
            for i, pv_potentials_month in enumerate(
                    pv_potentials_all_months_ev):
                # Generate random x-values centered around the box-plot.
                if type(pv_potentials_month) is pd.Series:
                    num_pts = len(pv_potentials_month)
                else:
                    num_pts = 1
                x = np.random.normal(loc=1 + i, scale=0.04,
                                     size=num_pts)
                plt.scatter(x, pv_potentials_month, alpha=0.1, color='C0')
        plt.savefig(fig_dir.joinpath(f'Charging_Potential_{ev_name}.svg'))
        plt.savefig(fig_dir.joinpath(f'Charging_Potential_{ev_name}.pdf'))
        pickle.dump(
            plt.gcf(),
            open(fig_dir.joinpath(
                f'Charging_Potential_{ev_name}.fig.pickle'), 'wb'))

    fig1 = plt.figure(figsize=figsize)
    # Seperate the `daily_pv_generated_p_month` dataframe into a list of
    # 12 dataframes, each representing a month of the year.
    date_index = pv_potentials_per_day.index.get_level_values('Date')
    pv_potentials_all_months = [
        pv_potentials_per_day.set_index([date_index.month]).\
        loc[j] for j in set(date_index.month)]
    pv_potentials_all_months = [
        pv_potentials_ev['PV_Charge_Pot_Per_m2'] / 3600000 for
        pv_potentials_ev in pv_potentials_all_months]
    plt.boxplot(pv_potentials_all_months,
                medianprops={'color': 'black'},
                flierprops={'marker': '.'},
                positions=list(set(date_index.month)))
    # box_stats = {  # XXX FIXME
    #     'box stats': [
    #         mpl.cbook.boxplot_stats(pv_potentials_month.values)[0] for
    #         pv_potentials_month in pv_potentials_all_months
    #     ]
    # }
    # with open(csv_dir.joinpath(f'box_pv_potentials.json'),
    #         'w') as f:
    #     json.dump(box_stats, f, cls=NumpyEncoder, indent=4)
    plt.ylabel('Charging potential per day (kWh/m2)', fontsize='small')
    plt.xlabel('Month of year')
    if plot_blotches:
        # Plot the stop_events which make up the box-plots.
        for i, pv_potentials_month in enumerate(
                pv_potentials_all_months_ev):
            # Generate random x-values centered around the box-plot.
            if type(pv_potentials_month) is pd.Series:
                num_pts = len(pv_potentials_month)
            else:
                num_pts = 1
            x = np.random.normal(loc=1 + i, scale=0.04,
                                 size=num_pts)
            plt.scatter(x, pv_potentials_month, alpha=0.1, color='C0')
    plt.tight_layout()

    fig2 = plt.figure(figsize=figsize)
    # Seperate the `daily_pv_generated_p_month` dataframe into a list of
    # 12 dataframes, each representing a month of the year.
    date_index = daily_pv_generated.index.get_level_values('Date')
    daily_pv_generated_all_months = [
        daily_pv_generated.set_index(date_index.month).loc[i] for i in
        range(1, 13)]
    daily_pv_generated_all_months = [
        daily_pv_generated['Daily_PV_Generation'] / 3600000 for
        daily_pv_generated in daily_pv_generated_all_months]
    plt.boxplot(daily_pv_generated_all_months,
                medianprops={'color': 'black'},
                flierprops={'marker': '.'},
                positions=list(set(date_index.month)))
    box_stats = {
        'box stats': [
            mpl.cbook.boxplot_stats(daily_pv_generated_month.values)[0] for
            daily_pv_generated_month in daily_pv_generated_all_months]}
    with open(csv_dir.joinpath('box_pv_generated.json'), 'w') as f:
        json.dump(box_stats, f, cls=NumpyEncoder, indent=4)
    plt.ylabel('PV energy generated per day (kWh/m2)', fontsize='small')
    plt.xlabel('Month of year')
    if plot_blotches:
        # Plot the stop_events which make up the box-plots.
        for i, daily_pv_generated in enumerate(daily_pv_generated_all_months):
            # Generate random x-values centered around the box-plot.
            x = np.random.normal(loc=1 + i, scale=0.04,
                                 size=len(daily_pv_generated))
            plt.scatter(x, daily_pv_generated, alpha=0.1, color='C0')
    plt.tight_layout()

    # Average energy charging potential for various months and various taxis.
    fig3 = plt.figure(figsize=figsize)
    width = 0.08
    for i, ev_name in enumerate(ev_names):
        df = pv_potentials_avg_of_month_avgs_per_taxi.loc[ev_name]
        plt.bar(df.index + (i - len(ev_names) / 2 - 0.5) * width,
                df['PV_Charge_Pot_Per_m2'] / 3600000, width=width)
    plt.legend(ev_names)
    plt.ylabel('Average energy in a day (kWh/m2)')
    plt.xlabel('Month of year')
    plt.tight_layout()

    # Save the plots
    fig_bar_chart.savefig(
        fig_dir.joinpath('dirty__monthly_charging_potential.png'))
    pickle.dump(
        fig_bar_chart,
        open(fig_dir.joinpath('dirty__monthly_charging_potential.fig.pickle'),
             'wb'))
    fig1.savefig(fig_dir.joinpath('dirty__monthly_charging_potential_box.png'))
    fig1.savefig(fig_dir.joinpath('dirty__monthly_charging_potential_box.pdf'))
    pickle.dump(
        fig1,
        open(fig_dir.joinpath(
             'dirty__monthly_charging_potential_box.fig.pickle'), 'wb'))
    fig2.savefig(fig_dir.joinpath('dirty__pv_energy_generated_box.png'))
    fig2.savefig(fig_dir.joinpath('dirty__pv_energy_generated_box.pdf'))
    pickle.dump(
        fig2,
        open(fig_dir.joinpath('dirty__pv_energy_generated_box.fig.pickle'),
             'wb'))
    fig3.savefig(
        fig_dir.joinpath('dirty__monthly_charging_potential_per_taxi.png'))
    fig3.savefig(
        fig_dir.joinpath('dirty__monthly_charging_potential_per_taxi.pdf'))
    pickle.dump(
        fig3,
        open(
            fig_dir.joinpath(
                'dirty__monthly_charging_potential_per_taxi.fig.pickle'),
            'wb'))

    auto_run = kwargs.get('auto_run', False)
    if not auto_run:
        plt.show()


if __name__ == "__main__":
    scenario_dir = Path(os.path.abspath(__file__)).parents[1]
    run_pv_results_analysis(scenario_dir)
