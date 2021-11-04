#!/usr/bin/env python3

"""
Extract power-energy plots from a bunch of pickled summary plots.

Instructions:
    Place this script in a directory which consists of pickled summary plots
    (`mean_plot.fig.pickle`s). Typically these plots will be from various
    scenarios which were simulated in a particular study. Run the script.
Inputs:
    Pickled summary plots:
        These plots should have been previously generated from the
        ev_results_analysis of ev-fleet-sim.
Output:
    Power-energy plots:
        Plots with the power and energy lines shown on one twin `Axes`. The
        Axes limits are chosen so that they are common across all the plots.
        I.e., the limits will be set from the maximum power and energy values
        encountered in the input plots.
Warnings:
    1:
        Please make sure that the version of matplotlib you are using when
        running this script is the same as the version used when the input
        summary plots were generated
"""

import pickle
import argparse
from data_processing_ev.results_analysis.ev_results_analysis import _y_fmt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=None)
    args = parser.parse_args()

    graphsdir = Path(__file__).parent
    outputdir = graphsdir.joinpath('power_energy')
    outputdir.mkdir(parents=True, exist_ok=True)

    # Load the fig.pickle files.
    if args.file:
        name = Path(args.file).stem
        figs = {name[:name.find('.fig')]:
                pickle.load(open(args.file, 'rb'))}
    else:
        figs = {file.stem[:file.stem.find('.fig')]:
                pickle.load(open(file, 'rb')) for file in
                graphsdir.glob('*.fig.pickle')}

    # Find the maximum and minimum limits of power and energy.
    power_min  = float('inf')
    power_max  = -float('inf')
    energy_min = float('inf')
    energy_max = -float('inf')
    for _, fig in figs.items():
        power_lims  = fig.get_axes()[0].get_ylim()
        energy_lims = fig.get_axes()[2].get_ylim()
        if power_lims[0] < power_min:
            power_min = power_lims[0]
        if power_lims[1] > power_max:
            power_max = power_lims[1]
        if energy_lims[0] < energy_min:
            energy_min = energy_lims[0]
        if energy_lims[1] > energy_max:
            energy_max = energy_lims[1]

    # Extract the components and save them.
    for name, fig in figs.items():
        # Add energy axis to power axes.
        power_ax = fig.get_axes()[0]
        energy_ax = fig.get_axes()[2]
        energy_twin_ax = power_ax.twinx()
        energy_data = energy_ax.lines[0].get_xydata()
        energy_twin_ax.plot(energy_data[:, 0], energy_data[:, 1],
                            linestyle='dashed', color='black',
                            label='Cumulative Energy')
        energy_twin_ax.yaxis.set_major_formatter(FuncFormatter(_y_fmt))

        energy_twin_ax.set_ylabel("Energy (Wh)")

        # Change scale of axes:
        power_ax.set_ylim(power_min, power_max)
        energy_twin_ax.set_ylim(energy_min, energy_max)

        # Add legend from both twin axes
        lines = [power_ax.lines[0], power_ax.lines[2],
                 energy_twin_ax.lines[0]]
        labels = [line.get_label() for line in lines]
        energy_twin_ax.legend(lines, labels, loc='upper left')

        power_ax.get_legend().remove()
        energy_twin_ax.get_legend().set_visible(False)

        # Save the power/energy twin axes.
        extent = fig.get_axes()[0].\
            get_window_extent().transformed(
                fig.dpi_scale_trans.inverted())
        fig.tight_layout()

        save_file = outputdir.joinpath(
            f'{name}_power_energy.pdf')
        fig.savefig(
            save_file, bbox_inches=extent.expanded(1.4, 1.26).\
            translated(-0.13, -0.32))

        save_file = outputdir.joinpath(
            f'{name}_power_energy.svg')
        fig.savefig(
            save_file, bbox_inches=extent.expanded(1.4, 1.26).\
            translated(-0.13, -0.32))

        save_file = outputdir.joinpath(
            f'{name}_power_energy.fig.pickle')
        pickle.dump(fig, open(save_file, 'wb'))
