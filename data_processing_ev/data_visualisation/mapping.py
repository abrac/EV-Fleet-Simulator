#!/usr/bin/env python3

import folium
from folium import plugins
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from typing import Iterable, Tuple


def gen_map_traces(scenario_dir: Path, mapping_points: bool = True,
                   mapping_lines: bool = False, mapping_heatmap: bool = True,
                   mapping_geojson: bool = False
                   ) -> Iterable[Tuple[str, folium.Map]]:
    """Map traces, save them & yield (trace_name, map)."""
    # TODO Read options from configuration file
    inputs_dir = scenario_dir.joinpath('_Inputs')
    scenario_name = scenario_dir.name
    # Get gps traces
    traces_dir = inputs_dir.joinpath('Traces', 'Processed')
    traces_list = [*traces_dir.glob('*.csv')]
    if len(traces_list) == 0:
        raise ValueError('No traces found in {traces_dir}.')
    # Get map features (taz boundaries, etc.)
    if mapping_geojson:
        map_dir = inputs_dir.joinpath('Map')
        geojson_filelist = [*map_dir.glob('*.geojson')]
        if len(geojson_filelist) == 1:
            geojson_file = geojson_filelist[0]
        else:
            raise ValueError(f'{map_dir} must contain *one* geojson file.')
        geojson_data = json.load(geojson_file)

    # Read location from the boundary.csv file.
    boundary_file = scenario_dir.joinpath('_Inputs', 'Map', 'Boundary',
                                          'boundary.csv')
    boundary = pd.read_csv(boundary_file)
    longitude = boundary.loc[:, 'Longitude'].mean()
    latitude = boundary.loc[:, 'Latitude'].mean()
    for trace_file in tqdm(traces_list):
        df = pd.read_csv(str(trace_file))
        # Initialise map
        map_area = folium.Map(location=[latitude, longitude],
                              titles=scenario_name, zoom_start=12)
        if mapping_geojson:
            # Add geographic data to map
            folium.GeoJson(geojson_data).add_to(map_area)
        if mapping_lines:
            folium.PolyLine(
                df[['Latitude', 'Longitude']].values).add_to(map_area)
        if mapping_points:
            # For each row in the traces dataset, plot the corresponding
            # latitude and longitude on the map.
            # TODO This is speed and memory inefficient.
            #      Maybe plot 1 in every 100 points, for example.
            df.apply(lambda row: folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]]).add_to(map_area),
                axis=1)
            # for i, row in tqdm(df.iterrows()):
            #     # if i == 0:
            #     #     folium.CircleMarker(
            #     #         (row.Latitude, row.Longitude), radius=10, weight=1,
            #     #         color='red', fill_color='red', fill_opacity=1
            #     #     ).add_to(map_area)
            #     # else:
            #     folium.CircleMarker([row['Latitude'],
            #                          row['Longitude']]).add_to(map_area)
        if mapping_heatmap:
            # Add heatmap
            map_area.add_child(plugins.HeatMap(
                data=df[['Latitude', 'Longitude']].values, radius=10, blur=7))
        yield (trace_file.stem, map_area)


def map_scenario(scenario_dir: Path, mapping_points: bool = False,
                 mapping_lines: bool = False, mapping_heatmap: bool = True,
                 mapping_geojson: bool = False, saving: bool = True
                 ) -> None:
    """Map traces and save them to the scenario directory."""
    scenario_name = scenario_dir.name
    for trace_name, map_trace in gen_map_traces(scenario_dir, mapping_points,
                                                mapping_lines, mapping_heatmap,
                                                mapping_geojson):
        # save the map as an html file
        output_file = scenario_dir.joinpath(
            'Data_Visualisations', 'Maps',
            '{0}_{1}.html'.format(scenario_name, trace_name))
        output_file.parent.mkdir(parents=True, exist_ok=True)
        map_trace.save(str(output_file))
