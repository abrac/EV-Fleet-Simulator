#!/usr/bin/env python3

import pickle
import argparse
from data_processing_ev.results_analysis.ev_results_analysis import _y_fmt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('fig_pickle_file')
args = parser.parse_args()

figx = pickle.load(open(args.fig_pickle_file, 'rb'))

graphsdir = Path(args.fig_pickle_file).parent

# extent = figx.get_axes()[0].\
#     get_window_extent().transformed(
#         figx.dpi_scale_trans.inverted())
# save_file = graphsdir.joinpath('total_power_profile.pdf')
# figx.savefig(
#     save_file, bbox_inches=extent.expanded(1.26, 1.26).\
#     translated(-0.35, -0.32))

# extent = figx.get_axes()[2].\
#     get_window_extent().transformed(
#         figx.dpi_scale_trans.inverted())
# save_file = graphsdir.joinpath('total_energy_profile.pdf')
# figx.savefig(
#     save_file, bbox_inches=extent.expanded(1.26, 1.26).\
#     translated(-0.35, -0.32))

# Add energy axis to power axes.
power_ax = figx.get_axes()[0]
energy_ax = figx.get_axes()[2]
energy_twin_ax = power_ax.twinx()
energy_data = energy_ax.lines[0].get_xydata()
energy_twin_ax.plot(energy_data[:, 0], energy_data[:, 1],
                    linestyle='dashed', color='black',
                    label='Cumulative Energy')
energy_twin_ax.yaxis.set_major_formatter(FuncFormatter(_y_fmt))

energy_twin_ax.set_ylabel("Energy (Wh)")

# Add legend from both twin axes
lines = [power_ax.lines[0], power_ax.lines[2],
         energy_twin_ax.lines[0]]
labels = [line.get_label() for line in lines]
power_ax.legend(lines, labels)

# Save the power/energy twin axes.
extent = figx.get_axes()[0].\
    get_window_extent().transformed(
        figx.dpi_scale_trans.inverted())
save_file = graphsdir.joinpath(
    'total_power_energy_profile.pdf')
figx.tight_layout()
figx.savefig(
    save_file, bbox_inches=extent.expanded(1.4, 1.26).\
    translated(-0.13, -0.32))
