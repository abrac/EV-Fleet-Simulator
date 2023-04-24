#! /usr/bin/env python3

""" This scripts generates box plots from the energy_usage csv files from GTFS
simulations of a set of cities. The script must be executed from a folder
containing multiple GTFS energy_usage csv files.  """

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

plt.rcParams['font.family'] = ['serif']
plt.rcParams["mathtext.fontset"] = 'cm'

if __name__ == "__main__":
    box_plots_dir = Path(__file__).parent
    city_energy_files = sorted([
        *box_plots_dir.glob('*.csv')
    ])

    city_names = [city_file.stem for city_file in city_energy_files]

    all_energy_diffs = []
    all_dists_travelled = []
    # For each city, read statistics and generate a box plot
    for city_energy_file in city_energy_files:
        city_energy_df = pd.read_csv(city_energy_file)
        city_energy_diffs = city_energy_df['energy_used']
        all_energy_diffs.append(city_energy_diffs)
        city_dist_travelled = city_energy_df['dist_travelled']
        all_dists_travelled.append(city_dist_travelled)
        kwh_p_km = (np.array(all_energy_diffs) /
                    (np.array(all_dists_travelled) / 1000))

    # Generate a box-plot from the values in kwh_p_km
    plt.figure(figsize=(5, 4))
    plt.boxplot(kwh_p_km,
                medianprops={'color': 'black'},
                flierprops={'marker': '.'},
                showmeans=True)
    plt.ylabel("Energy consumption (kWh/km)")
    plt.xticks(range(1, len(city_names) + 1), city_names, rotation=20,
               fontsize='small')
    plt.xlabel("Scenario")
    plt.tight_layout()

    plt.ylim(0,1)

    # As png:
    fig_dir = box_plots_dir.joinpath("Energy_usage_box_plots.png")
    plt.savefig(fig_dir)
    # As svg:
    fig_dir = box_plots_dir.joinpath("Energy_usage_box_plots.pdf")
    plt.savefig(fig_dir)
    # As pickle:
    fig_dir = box_plots_dir.joinpath(
        "Energy_usage_box_plots.fig.pickle")
    fig = plt.gcf()
    pickle.dump(fig, open(fig_dir, 'wb'))

    plt.show()
