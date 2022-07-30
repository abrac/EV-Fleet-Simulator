#!/usr/bin/env python3

import os
import folium
from folium import plugins
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Iterable, Generator
import data_processing_ev as dpr


def _reformat_time(time_old, **kwargs):
    # time format: "year/month/day hr12:min:sec meridiem" -->
    # "year-month-day{T}hr24:min:sec"
    time_old = str(time_old)
    input_data_fmt = kwargs.get('input_data_fmt')
    if input_data_fmt == dpr.DATA_FMTS['GPS']:
        (date, time) = time_old.split(' ')
        (year, month, day) = date.split("-")
        (hr24_str, minute, sec) = time.split(":")
        hr24 = int(hr24_str)
        # hr24 = (int(hr12) % 12) + (12 if meridiem == "PM" else 0)
        return_str = f"{year}-{month}-{day}T{hr24:02d}:{minute}:{sec}"
    elif input_data_fmt == dpr.DATA_FMTS['GTFS']:
        (date, time) = time_old.split(' ')
        (year, month) = ['2000', '01']
        day = int(date[:-1]) + 1
        (hr24_str, minute, sec) = time.split(":")
        hr24 = int(hr24_str)
        return_str = f"{year}-{month}-{day:02d}T{hr24:02d}:{minute}:{sec}"
    return return_str


def _rainbowfy(speed, max_speed):
    """
    Parameters:
    speed (float)
    max_speed (float)

    Returns:
    string: Color of rainbow corresponding to number
    """
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    return colors[int(speed/max_speed*len(colors))] if speed < max_speed \
        else colors[-1]


def animate_route(df, map, **kwargs):
    """
    Animates route followed by coordinates of `df`
    """
    # FIXME This is memory inefficient for large files.
    lines = []
    dfiterator = df.iterrows()
    first_iter = True
    for _, row in dfiterator:
        if first_iter:
            coord2_lat = row.Latitude
            coord2_lon = row.Longitude
            date2 = _reformat_time(row.Time, **kwargs)
            first_iter = False
            continue

        coord1_lat = coord2_lat
        coord1_lon = coord2_lon
        date1 = date2
        coord2_lat = row.Latitude
        coord2_lon = row.Longitude
        date2 = _reformat_time(row.Time, **kwargs)
        colour = _rainbowfy(row.Velocity, max_speed=100)

        lines.append(
            {
                'coordinates': [
                    [coord1_lon, coord1_lat],
                    [coord2_lon, coord2_lat],
                ],
                'dates': [
                    date1,
                    date2
                ],
                'color': colour
            }
        )

    features = [
        {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': line['coordinates'],
            },
            'properties': {
                'times': line['dates'],
                'style': {
                    'color': line['color'],
                    'weight': line['weight'] if 'weight' in line else 5
                }
            }
        }
        for line in lines
    ]

    plugins.TimestampedGeoJson({
        'type': 'FeatureCollection',
        'features': features,
    }, period='PT1M', add_last_point=False, time_slider_drag_update=True) \
        .add_to(map)


def gen_scenario_animations(scenario_dir: Path,
                            saving: bool = True, **kwargs) -> Iterable[folium.Map]:
    traces_dir = scenario_dir.joinpath('_Inputs', 'Traces', 'Processed')
    save_dir = scenario_dir.joinpath('Data_Visualisations',
                                     'Route_Animations')
    traces_list = traces_dir.glob('*.csv')
    scenario_name = scenario_dir.absolute().name

    # Read location from the boundary.csv file.
    boundary_file = scenario_dir.joinpath('_Inputs', 'Map', 'Boundary',
                                          'boundary.csv')
    boundary = pd.read_csv(boundary_file, skipinitialspace=True,
                           sep='\\s*,\\s*', engine='python')
    longitude = boundary.loc[:, 'Longitude'].mean()
    latitude = boundary.loc[:, 'Latitude'].mean()
    for trace_file in tqdm(traces_list):
        trace_savedir = save_dir.joinpath(trace_file.stem)
        trace_savedir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(str(trace_file))
        df['Time'] = pd.to_datetime(df['Time'])

        for date, sub_df in df.groupby([df['Time'].dt.date]):
            # Initialise map around Stellenbosch
            map_area = folium.Map(location=[latitude, longitude],
                                  titles=scenario_name, zoom_start=14)

            animate_route(sub_df, map_area, **kwargs)

            # save the map as an html
            if saving:
                save_file = trace_savedir.joinpath(
                    '{0}_{1}_{2}.html'.format(scenario_name,
                                              trace_file.stem,
                                              str(date)))
                map_area.save(str(save_file))

            yield map_area


def animate_scenario(scenario_dir: Path, saving: bool = True,
                     **kwargs) -> None:
    for _ in gen_scenario_animations(scenario_dir, saving, **kwargs):
        pass
