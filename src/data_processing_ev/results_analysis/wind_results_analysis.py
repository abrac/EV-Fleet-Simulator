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


def _load_windspeed_data(scenario_dir: Path, year: int) -> tp.Callable:
    # Load irradiance data as dataframe
    windspeed_file = scenario_dir.joinpath('SAM_Simulation', 'Results',
                                            'POA_windspeed', 'Data',
                                            f'{year}.csv')
        # TODO Don't rely on just one year, maybe take the average across
        # multiple years... And maybe keep each month seperate.
    wind_output_df = pd.read_csv(windspeed_file)
    wind_output_df.columns = ['DateTime', 'P_Out']

    # Convert first column to datetime objects.
    wind_output_df['DateTime'] = (
        pd.Timestamp(year=year, month=1, day=1, hour=0, minute=0) +
        pd.to_timedelta(wind_output_df['DateTime'], unit='H'))
    wind_output_df = wind_output_df.set_index('DateTime')

    # Create a function which interpolates between points of wind_output_df. This
    # will be used for integration.
    f_wind_out = interpolate.interp1d(
        x=pd.Series(wind_output_df.index).apply(
            lambda datetime: time.mktime(datetime.timetuple())),
        y=wind_output_df['P_Out'],
        fill_value='extrapolate')

    return f_wind_out


def run_wind_results_analysis(scenario_dir: Path, plot_blotches: bool = False,
                            figsize=(4, 3), **kwargs):
    input_data_fmt = kwargs.get('input_data_fmt', dpr.DATA_FMTS['GPS'])
    # Load windspeed data
    # FIXME Don't hardcode this. Don't even ask the year.
    _ = input("For which year is the wind speed data? "
              "(Leave blank for 2017.) \n\t")
    try:
        year = int(_)
    except ValueError:
        year = 2017

    # Directories for saving CSVs and figures.
    csv_dir = scenario_dir.joinpath('SAM_Simulation', 'csvs')
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = scenario_dir.joinpath('SAM_Simulation', 'graphs')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load the windspeed function.
    f_wind_out = _load_windspeed_data(scenario_dir, year)

    # Load list of stop-arrivals and -durations as DataFrame.
    stops_file = scenario_dir.joinpath('Temporal_Clusters', 'Clustered_Data',
                                       'time_clusters_with_dates.csv')
    stops_df = pd.read_csv(stops_file).set_index(['EV_Name', 'Date'])

    # Check if wind_potentials has been calculated already.
    wind_potentials_file = csv_dir.joinpath('wind_potentials.csv')
    use_existing_wind_file = False
    # If so, ask if loading it.
    if wind_potentials_file.exists():
        _ = input(f"wind_potentials.csv found at: \n\t {wind_potentials_file} \n" +
                  "Use this file? [Y/n]")
        use_existing_wind_file = True if _.lower() != 'n' else False

    # If loading it, load it, else regenerate wind_potentials.
    if use_existing_wind_file:
        wind_potentials = pd.read_csv(wind_potentials_file)
    else:
        # Integrate all the energy charged at all stopped times.
        wind_potentials = []
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
            stop_beg = [int(stop_beg_float), int((stop_beg_float % 1) * 60)]
            stop_beg = dt.datetime(date.year, date.month, date.day, *stop_beg)
            stop_beg = time.mktime(stop_beg.timetuple())

            # Get the end of the stop.
            stop_end_float = stop_beg_float + stop['Stop_Duration']
            stop_end = [int(stop_end_float), int((stop_end_float % 1) * 60)]
            # Handle case where stop extends to the next day:
            while stop_end[0] >= 24:
                stop_end[0] -= 24
                date += dt.timedelta(1)
            stop_end = dt.datetime(date.year, date.month, date.day, *stop_end)
            stop_end = time.mktime(stop_end.timetuple())

            # Integrate to find the wind energy for this stop-event.
            stop_wind_potential, error = integrate.quad(f_wind_out, stop_beg,
                                                      stop_end)

            # Append the results of the integration to the result-lists.
            integration_errors.append(error)
            stop_copy = stop.copy()
            stop_copy['wind_Charge_Pot'] = stop_wind_potential
            wind_potentials.append(stop_copy)

        # Create a warning if there are discarded stops.
        if long_stops_count:
            logging.warn(f'Stops discarded: {long_stops_count}')
            print(f'Stops discarded: {long_stops_count}')
        else:
            print("No stops discarded.")
        # Convert the wind_potentials list into a dataframe.
        wind_potentials = pd.DataFrame(wind_potentials)
        wind_potentials.index = pd.MultiIndex.from_tuples(wind_potentials.index)
        wind_potentials.index.names = ['EV_Name', 'Date']
        wind_potentials = wind_potentials.reset_index()
        # Save the wind_potentials dataframe.
        wind_potentials_file.parent.mkdir(parents=True, exist_ok=True)
        wind_potentials.to_csv(wind_potentials_file, index=False)

    # Make date index-level consist of date-time objects.
    wind_potentials['Date'] = pd.to_datetime(wind_potentials.reset_index()['Date'])
    wind_potentials = wind_potentials.set_index(['EV_Name', 'Date'])

    # For each date, sum up the total charging potential for that date.
    wind_potentials_per_day = wind_potentials.\
        reset_index().groupby(['EV_Name', 'Date']).\
        sum()[['Stop_Duration', 'wind_Charge_Pot']]

    # For each taxi, fill missing dates with charging potentials of zero.
    # Read csv file with filtered dates.
    filtered_dates = pd.read_csv(
        scenario_dir.joinpath('Spatial_Clusters', 'Filtered_Traces',
                              'dirty__dates_remaining.csv')
    )
    # Convert the dates to date-time objects, and set it as the index.
    filtered_dates['Date'] = pd.to_datetime(filtered_dates['Date'])
    filtered_dates[['Stop_Duration', 'wind_Charge_Pot']] = 0
    filtered_dates = filtered_dates.set_index(['EV_Name', 'Date'])
    # Merge into `wind_potentials_per_day` the dates which are not in
    #   `wind_potentials_per_day`, but are in `filtered_dates`.
    wind_potentials_per_day = pd.concat([
        wind_potentials_per_day,
        filtered_dates[~filtered_dates.index.isin(wind_potentials_per_day.index)]
    ]).sort_index()

    # TODO Remove the below. We are no longer going to fill all missing dates.
    """
    ev_names = sorted(set(
        wind_potentials_per_day.index.get_level_values('EV_Name')
    ))
    # list of ev_dfs:
    ev_dfs = []
    for ev_name in ev_names:
        df = wind_potentials_per_day.loc[ev_name]
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
    wind_potentials_per_day = pd.concat(
        ev_dfs, keys=ev_names, names=['EV_Name', 'Date']
    )
    """

    # Get average daily wind potential over each month.
    date_index = wind_potentials_per_day.index.get_level_values('Date')
    ev_name_index = wind_potentials_per_day.index.get_level_values('EV_Name')
    _ = input("""Would you like to obtain the monthly average by adding the
                 values and dividing by dates on record, or by dividing by
                 dates in month? [RECORD/month] \n\t""")
    divide_by_days_in_month = False if _.lower() != 'month' else True
    if divide_by_days_in_month:
        # Get the sum of the wind potentials for each month.
        wind_potentials_per_month = wind_potentials_per_day.groupby(
            [ev_name_index, date_index.year, date_index.month]
        ).sum()
        wind_potentials_per_month.index.names = ['EV_Name', 'Year', 'Month']
        # Initialise a list, in which to append Sieries's.
        wind_potentials_month_list = []
        for _, wind_potential_month in wind_potentials_per_month.iterrows():
            # Number of days in month.
            curr_year = wind_potential_month.name[1]
            curr_month = wind_potential_month.name[2]
            days_in_month = calendar.monthrange(curr_year, curr_month)[1]
            # Divide total energy by number of days in that month.
            average_potential = \
                wind_potential_month['wind_Charge_Pot'] / days_in_month
            average_stop = \
                wind_potential_month['Stop_Duration'] / days_in_month
            # Create a Series (row of DataFrame) with the value, and append to
            # list.
            row = pd.Series(
                {
                    'Stop_Duration': average_stop,
                    'wind_Charge_Pot': average_potential},
                name=wind_potential_month.name)
            wind_potentials_month_list.append(row)
        # Convert the list to a DataFrame.
        wind_potentials_month_avg = pd.DataFrame(
            wind_potentials_month_list,
            pd.MultiIndex.from_tuples(
                [row.name for row in wind_potentials_month_list],
                names=['EV_Name', 'Year', 'Month']
            )
        )
    else:
        wind_potentials_month_avg = wind_potentials_per_day.groupby(
            [ev_name_index, date_index.year, date_index.month]
        ).mean()
        wind_potentials_month_avg.index.names = ['EV_Name', 'Year', 'Month']

    # Get average energy of months across years:
    wind_potentials_avg_of_month_avgs_per_taxi = wind_potentials_month_avg.\
        reset_index().groupby(['EV_Name', 'Month']).\
        mean()[['Stop_Duration', 'wind_Charge_Pot']]  # TODO Bad variable name.

    # Get average energy of months across years and across taxis:
    wind_potentials_avg_of_month_avgs = wind_potentials_month_avg.\
        reset_index().groupby(['Month']).\
        mean()[['Stop_Duration', 'wind_Charge_Pot']]

    # Get approximate monthly total energy from wind:
    wind_potentials_total_per_month = wind_potentials_avg_of_month_avgs.copy()
    wind_potentials_total_per_month['wind_Charge_Pot'] *= 30

    # Get *total* wind energy generated by the plane
    # (not only energy used for charging EV):
    daily_wind_generated_p_month = []
    integration_errors = []
    for month in range(1, 13):
        days_in_month = calendar.monthrange(year, month)[1]
            # FIXME Warning: the year used here is different than the year
            # used to calculate the days_in_month for the
            # wind_charging_potential. Do a reverse search of "days_in_month" to
            # find the section that I am talking about.
        start = time.mktime(dt.datetime(year, month, 1, 0).timetuple())
        end = time.mktime(dt.datetime(year, month, days_in_month, 23, 59, 59
                                      ).timetuple())
        month_wind_generated, error = integrate.quad(f_wind_out, start, end)
        daily_wind_generated_p_month.append(month_wind_generated / days_in_month)
        integration_errors.append(error / days_in_month)
    daily_wind_generated_p_month = pd.DataFrame(
        daily_wind_generated_p_month, index=range(1, 13),
        columns=['Avg Daily wind Generation'])
    daily_wind_generated_p_month.index.names = ['Month']

    # Check if wind_potentials has been calculated already.
    wind_generated_file = csv_dir.joinpath('daily_wind_generated.csv')
    use_existing_wind_file = False
    # If so, ask if loading it.
    if wind_generated_file.exists():
        _ = input("daily_wind_generated.csv found at: \n\t " +
                  f"{wind_generated_file} \n Use this file? [Y/n]")
        use_existing_wind_file = True if _.lower() != 'n' else False

    # If loading it, load it, else regenerate wind_potentials.
    if use_existing_wind_file:
        daily_wind_generated = pd.read_csv(wind_generated_file)
        daily_wind_generated = daily_wind_generated.set_index(['Date'])
    else:
        daily_wind_generated = []
        integration_errors = []
        days = 367 if calendar.isleap(year) else 366
        for day in tqdm(range(1, days)):
                # FIXME Warning: the year used here is different than the year
                # used to calculate the days_in_month for the
                # wind_charging_potential. Do a reverse search of "days_in_month" to
                # find the section that I am talking about.
            beginning_of_year = dt.datetime(year, 1, 1, 0)
            start_date = beginning_of_year + dt.timedelta(day - 1)
            start = time.mktime(start_date.timetuple())
            end = time.mktime(
                dt.datetime(start_date.year, start_date.month, start_date.day,
                            23, 59, 59).timetuple())
            day_wind_generated, error = integrate.quad(f_wind_out, start, end)
            day_wind_generated = pd.Series(
                day_wind_generated, index=['Daily_wind_Generation'],
                name=start_date.date())
            daily_wind_generated.append(day_wind_generated)
            integration_errors.append(error)

        daily_wind_generated = pd.DataFrame(daily_wind_generated,
                                          columns=['Daily_wind_Generation'])
        daily_wind_generated.index.names = ['Date']
        daily_wind_generated.to_csv(wind_generated_file)

    daily_wind_generated.index = pd.to_datetime(daily_wind_generated.index)

    # Save csv files of the dataframes for future reference.
    wind_potentials_per_day.to_csv(
        csv_dir.joinpath('wind_potentials_per_day.csv'))
    wind_potentials_month_avg.to_csv(
        csv_dir.joinpath('wind_potentials_month_avg.csv'))
    wind_potentials_avg_of_month_avgs.to_csv(
        csv_dir.joinpath('wind_potentials_avg_of_month_avgs.csv'))
    wind_potentials_avg_of_month_avgs_per_taxi.to_csv(
        csv_dir.joinpath('wind_potentials_avg_of_month_avgs_per_taxi.csv'))
    daily_wind_generated_p_month.to_csv(
        csv_dir.joinpath('daily_wind_generated_p_month.csv'))

    # Generate bar plots of the charging potential.
    # Average energy charging potential for various months of the year.
    fig_bar_chart = plt.figure(figsize=figsize)
    width = 0.3
    plt.bar(wind_potentials_avg_of_month_avgs.index - width / 2,
            wind_potentials_avg_of_month_avgs['wind_Charge_Pot'] / 3600,
            width=width)
    plt.bar(daily_wind_generated_p_month.index + width / 2,
            daily_wind_generated_p_month['Avg Daily wind Generation'] / 3600,
            width=width)
    plt.ylabel('Average energy in a day (kWh)') 
    plt.xlabel('Month of year')
    plt.legend(['Energy charged directly from wind turbine',
                'Total wind energy generated'])
    plt.tight_layout()

    # Plot box-plot version of the above figure.
    ev_names = sorted(set(
        wind_potentials_avg_of_month_avgs_per_taxi.index.\
        get_level_values('EV_Name')))
    for i, ev_name in enumerate(ev_names):
        # plt.subplot(3, 3, i+1)  # FIXME: Uncomment me!
        plt.figure(figsize=(3, 2))  # FIXME: Remove me!
        # Separate the `wind_potentials_avg_of_month_avgs` dataframe into a list
        # of 12 dataframes, each representing a month of the year.
        date_index = wind_potentials_per_day.loc[ev_name].index.get_level_values('Date')
        wind_potentials_all_months = [
            wind_potentials_per_day.loc[ev_name].set_index([date_index.month]).\
            loc[j] for j in set(date_index.month)]

        wind_potentials_all_months_ev = []
        for wind_potentials_ev in wind_potentials_all_months:
            if type(wind_potentials_ev) is pd.DataFrame:
                wind_potential = wind_potentials_ev['wind_Charge_Pot'] / 3600
            else:
                wind_potentials_ev = wind_potentials_ev.to_frame().T
                wind_potential = wind_potentials_ev['wind_Charge_Pot'] / 3600
            wind_potentials_all_months_ev.append(wind_potential)

        plt.boxplot(wind_potentials_all_months_ev,
                    medianprops={'color': 'black'},
                    positions=list(set(date_index.month)),
                    flierprops={'marker': '.'})
        box_stats = {
            'box stats': [
                mpl.cbook.boxplot_stats(wind_potentials_ev.values)[0] for
                wind_potentials_ev in wind_potentials_all_months_ev]}
        with open(csv_dir.joinpath(f'box_wind_potential_{ev_name}.json'),
                  'w') as f:
            json.dump(box_stats, f, cls=NumpyEncoder, indent=4)

        # plt.ylabel('Energy charged in a day (kWh)')  # FIXME: Uncomment
        # plt.xlabel('Month of year')  # FIXME: Uncomment
        # plt.title(ev_name)  # FIXME: Uncomment
        # plt.xticks(range(1,13), range(1,13))
        plt.ylim((-0.06, 1.3))
        # plt.xlim((0.5, 12.5))
        plt.tight_layout()
        if (plot_blotches):
            # Plot the stop_events which make up the box-plots.
            for i, wind_potentials_month in enumerate(
                    wind_potentials_all_months_ev):
                # Generate random x-values centered around the box-plot.
                if type(wind_potentials_month) is pd.Series:
                    num_pts = len(wind_potentials_month)
                else:
                    num_pts = 1
                x = np.random.normal(loc=1 + i, scale=0.04,
                                     size=num_pts)
                plt.scatter(x, wind_potentials_month, alpha=0.1, color='C0')
        plt.savefig(fig_dir.joinpath(f'Charging_Potential_{ev_name}.svg'))
        plt.savefig(fig_dir.joinpath(f'Charging_Potential_{ev_name}.pdf'))
        pickle.dump(
            plt.gcf(),
            open(fig_dir.joinpath(
                f'Charging_Potential_{ev_name}.fig.pickle'), 'wb'))

    fig1 = plt.figure(figsize=figsize)
    # Seperate the `daily_wind_generated_p_month` dataframe into a list of
    # 12 dataframes, each representing a month of the year.
    date_index = wind_potentials_per_day.index.get_level_values('Date')
    wind_potentials_all_months = [
        wind_potentials_per_day.set_index([date_index.month]).\
        loc[j] for j in set(date_index.month)]
    wind_potentials_all_months = [
        wind_potentials_ev['wind_Charge_Pot'] / 3600 for
        wind_potentials_ev in wind_potentials_all_months]
    plt.boxplot(wind_potentials_all_months,
                medianprops={'color': 'black'},
                flierprops={'marker': '.'},
                positions=list(set(date_index.month)))
    # box_stats = {  # XXX FIXME
    #     'box stats': [
    #         mpl.cbook.boxplot_stats(wind_potentials_month.values)[0] for
    #         wind_potentials_month in wind_potentials_all_months
    #     ]
    # }
    # with open(csv_dir.joinpath(f'box_wind_potentials.json'),
    #         'w') as f:
    #     json.dump(box_stats, f, cls=NumpyEncoder, indent=4)
    plt.ylabel('Charging potential per day (kWh)', fontsize='small') 
    plt.xlabel('Month of year')
    if plot_blotches:
        # Plot the stop_events which make up the box-plots.
        for i, wind_potentials_month in enumerate(
                wind_potentials_all_months_ev):
            # Generate random x-values centered around the box-plot.
            if type(wind_potentials_month) is pd.Series:
                num_pts = len(wind_potentials_month)
            else:
                num_pts = 1
            x = np.random.normal(loc=1 + i, scale=0.04,
                                 size=num_pts)
            plt.scatter(x, wind_potentials_month, alpha=0.1, color='C0')
    plt.tight_layout()

    fig2 = plt.figure(figsize=figsize)
    # Seperate the `daily_wind_generated_p_month` dataframe into a list of
    # 12 dataframes, each representing a month of the year.
    date_index = daily_wind_generated.index.get_level_values('Date')
    daily_wind_generated_all_months = [
        daily_wind_generated.set_index(date_index.month).loc[i] for i in
        range(1, 13)]
    daily_wind_generated_all_months = [
        daily_wind_generated['Daily_wind_Generation'] / 3600 for
        daily_wind_generated in daily_wind_generated_all_months]
    plt.boxplot(daily_wind_generated_all_months,
                medianprops={'color': 'black'},
                flierprops={'marker': '.'},
                positions=list(set(date_index.month)))
    box_stats = {
        'box stats': [
            mpl.cbook.boxplot_stats(daily_wind_generated_month.values)[0] for
            daily_wind_generated_month in daily_wind_generated_all_months]}
    with open(csv_dir.joinpath('box_wind_generated.json'), 'w') as f:
        json.dump(box_stats, f, cls=NumpyEncoder, indent=4)
    plt.ylabel('wind energy generated per day (kWh)', fontsize='small') 
    plt.xlabel('Month of year')
    if plot_blotches:
        # Plot the stop_events which make up the box-plots.
        for i, daily_wind_generated in enumerate(daily_wind_generated_all_months):
            # Generate random x-values centered around the box-plot.
            x = np.random.normal(loc=1 + i, scale=0.04,
                                 size=len(daily_wind_generated))
            plt.scatter(x, daily_wind_generated, alpha=0.1, color='C0')
    plt.tight_layout()

    # Average energy charging potential for various months and various taxis.
    fig3 = plt.figure(figsize=figsize)
    width = 0.08
    for i, ev_name in enumerate(ev_names):
        df = wind_potentials_avg_of_month_avgs_per_taxi.loc[ev_name]
        plt.bar(df.index + (i - len(ev_names) / 2 - 0.5) * width,
                df['wind_Charge_Pot'] / 3600, width=width)
    plt.legend(ev_names)
    plt.ylabel('Average energy in a day (kWh)') 
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
    fig2.savefig(fig_dir.joinpath('dirty__wind_energy_generated_box.png'))
    fig2.savefig(fig_dir.joinpath('dirty__wind_energy_generated_box.pdf'))
    pickle.dump(
        fig2,
        open(fig_dir.joinpath('dirty__wind_energy_generated_box.fig.pickle'),
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

    plt.show()


if __name__ == "__main__":
    scenario_dir = Path(os.path.abspath(__file__)).parents[1]
    run_wind_results_analysis(scenario_dir)
