#!/usr/bin/env python3

import os
import sys
import itertools  # noqa
import numpy as np  # noqa
from pathlib import Path
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt  # noqa

if "SUMO_HOME" in os.environ:
    xml2csv = Path(os.environ["SUMO_HOME"], "tools", "xml", "xml2csv.py")
else:
    sys.exit("Please declare environmental variable 'SUMO_HOME'.")

# plt.ion()
plt.style.use('default')


# %% Methods ##################################################################
# Importing plotting function from the *original* data_analysis script
main_data_analysis = Path(os.environ["ETAXI_HOME"], "Experiments",
                          "Data_Processing_Taxi", "6_Data_Analysis")
sys.path.insert(1, str(main_data_analysis))
from data_analysis import Data_Analysis  # noqa
# data_analysis.py: ../6_Data_Analysis/data_analysis.py


# %% Main #####################################################################
if __name__ == '__main__':
    filedir = Path(__file__).parent
    POPULATIONS = [0, 1000, 10000, 20000, 30000, 50000, 70000]

    # %% Get list of all battery input directories of all the experiments
    ev_day_dirs = [ev_xml.parent for
                   ev_xml in filedir.glob('Taxi_Routes/*/*/0/Battery.out.xml')]
    indiv_ev_dirs = set(ev_xml.parents[2] for ev_xml in
                        filedir.glob('Taxi_Routes/*/*/0/Battery.out.xml'))
    agg_ev_dir = filedir.joinpath('Taxi_Routes')

    data_analysis = Data_Analysis(agg_ev_dir, indiv_ev_dirs, ev_day_dirs)
    # data_analysis.make_plots()
    data_analysis.save_ev_stats()
    data_analysis.save_fleet_stats()
