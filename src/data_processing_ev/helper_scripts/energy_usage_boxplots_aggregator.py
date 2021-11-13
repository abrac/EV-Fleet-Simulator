#! /usr/bin/env python3

""" This scripts generates box plots from the energy_usage csv files from GTFS
simulations of a set of cities. The script must be executed from a folder
containing multiple GTFS energy_usage csv files.  """

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import pickle

if __name__ == "__main__":
    box_plots_dir = Path(__file__).parent
    city_energy_files = sorted([
        *box_plots_dir.glob('*.csv')
    ])

    city_names = [city_file.stem for city_file in city_energy_files]

    all_energy_diffs = []
    # For each city, read statistics and generate a box plot
    for city_energy_file in city_energy_files:
        city_energy_diffs = pd.read_csv(city_energy_file)['energy_used']
        all_energy_diffs.append(city_energy_diffs)

    # Generate a box-plot from the values in all_energy_diffs.
    plt.figure(figsize=(4, 3))
    plt.boxplot(all_energy_diffs,
                medianprops={'color': 'black'},
                flierprops={'marker': '.'})
    plt.ylabel("Daily energy usage (kWh)")
    plt.xticks(range(1, len(city_names) + 1), city_names, rotation=30,
               fontsize='small')
    plt.xlabel("City")
    plt.tight_layout()

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
