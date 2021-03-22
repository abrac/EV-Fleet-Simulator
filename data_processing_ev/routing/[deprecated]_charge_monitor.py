#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% [markdown]
# Importing modules

# %%
import traci.constants as tc
import traci
import os
import sys

# %% [markdown]
# # TraCI Implementation
# Appending Sumo `tools` folder (contains TraCI) to the PATH environment
# variable:

# %%
# Constants
SUMOCFG_PATH_HARDCODED = r"./Taxi_Routes/" + \
    r"osm_out_T1000_20131101_20151031.trip.xml/iteration_000.sumocfg"
WORKING_DIRECTORY = os.path.dirname(__file__)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environmental variable 'SUMO_HOME'.")

# %% [markdown]
# TO DO:
#
# 1. Invoke PySAM or SSC Library (see SAM code generator for example) and
# import SAM simulation results.
# 2. Modify `sumocfg` file or `taxi.trips.xml` file according to the
# results (if applicable)

# %% [markdown]
# Importing `sumocfg` and `Sumo.exe` directories:

# %%
input_sumo_gui = input("Run sumo with gui? ([y]/n)\n\t")
sumo_executable = "sumo" if input_sumo_gui.capitalize() == 'N' else "sumo-gui"
sumoBinary = os.path.join(os.environ["SUMO_HOME"], "bin", sumo_executable)
input_sumocfg_path = input(
    "Enter path to sumocfg file [leave blank to use hard-coded path]: \n\t")
sumocfg_path = os.path.join(
    SUMOCFG_PATH_HARDCODED) if input_sumocfg_path == "" else (
        os.path.join(input_sumocfg_path))
sumoCmd = [sumoBinary, "-c", sumocfg_path]

# %% [markdown]
# Starting TraCI and Sumo:

# %%
if input_sumo_gui.upper() != "N":
    print("Waiting for user to run simulation...")
traci.start(sumoCmd)
# while traci.simulation_isLoaded() is True: pass

# %% [markdown]
# # Specific implementation

# %%
# We are going to track taxi_0
TRACKED_VEHICLE = "T1000_20131101_20151031"
# Subscribing to variables important to TRACKED_VEHICLE
traci.vehicle.subscribe(
    TRACKED_VEHICLE,
    [tc.VAR_PERSON_NUMBER, tc.VAR_SPEED, tc.VAR_ACCELERATION,
     tc.VAR_ELECTRICITYCONSUMPTION])
for step in range(10000000):
    print("Step:", step)
    try:
        print(traci.vehicle.getSubscriptionResults(TRACKED_VEHICLE))
        print(traci.vehicle.getParameter(
            TRACKED_VEHICLE,
            "device.battery.actualBatteryCapacity"))
    except traci.TraCIException:
        print()
    traci.simulationStep()
traci.close()  # waits until Sumo process is finished, then closes.

# %% [Markdown]
# The following code implements a `StepListener` child class

# %%
# class ExampleListener(traci.StepListener):
#     def step(self, t=0):
#         print("ExampleListener called at %s ms." % t)
#         return True # indicate that the step listener should remain active in
#         # the next step
#         return True
# listener = ExampleListener()
# traci.addStepListener(listener)
