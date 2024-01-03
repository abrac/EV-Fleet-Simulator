<!-- Note: This is a markdown file. Use a markdown editor to easily edit and
     view this file. Just search the web for a nice markdown editor (like
     Ghostwriter). -->

- [ ] Critical!!!! Make sure the "monolithic" folder does not get copied during the splitting process for GTFS EV outputs.
- [ ] Create a fork of gtfs2gps at version 0.6-0
- [ ] Create a script which automatically calls GTFS_Convert, calls GTFS_Split, and extracts the GTFS.zip file.
- [ ] Test SUMO 1.12, so that it can be automatically installed through pip.
- [ ] Add WSL instructions for Windows. 
- [ ] Add RAM requirements to docs.
- Automate step 0 (scenario initialisation):
    - [ ] Merge the `get_max_min_coords()` funtion of `src/data_processing_ev/scenario_initialisation/Data_Pre-Processing/max_min_coords.py` into the `scenario_initialisation` module, such that step 0 runs it if there are traces found. If there are no traces in the directory (as should be the case when running step 0 initially), then it shouldn't run `get_max_min_coords()`.
    - [ ] Make step 0 prompt for the desired min and max coordinates, so that it automatically populates the values in `boundary.csv` and `pbf_to_osm.[bat|sh]`.
    - [ ] Reflect in the documentation that both of these things need to happen by running step 0 twice.
    - [ ] Get the mean location of the dataset using `get_max_min_coords()`. Prompt the user to use this location or another user-specified location for downloading the weather data (solar) automatically from SAM. Update documentation accordingly.
- [ ] When calculating average energy consumption per kilometer, divide the total energy consumption by the total distance travelled (across _all_ days).
- [ ] Simulate all vehicles _concurrently_ on one map, to hopefully improve simulation time!!!
- [ ] Standardise input file format to use m/s as it's velocity unit, rather than km/h. SI units, please!
- [ ] Update the docs to describe what inputs are required, in various simulation configurations.
- [ ] Merge [fcv-fleet-sim](https://gitlab.com/eputs/fcv-fleet-sim) FCV simulation model.
- [ ] Make `^C` cancel more elegantly. Currently it throws useless debugging info.
- [ ] Improve SAM Simulation documentation!
- [ ] Create a sub-module which does battery-sizing estimates.
- [ ] Add stop-location functionality.

- [o] (De-)Compressing those battery.out.csv files one by one is too slow. Perhaps we can rather do them all in bulk at the beginning and end of the data analysis script... It is also impacting the speed fo Hull's EV model. [Cancelled: I don't observe this anymore.]
- [o] Automatically run steps with a options given in a config file. [Cancelled: There are not many options to be configured. Those options can be added as command line flags.]
- [o] Create a setup script which automatically downloads SUMO, ev-fleet-sim, and also runs activate-global-python-argcomplete

- [x] Make the HULL model use **backward** differences by default.
- [x] Make the Hull simulation multithreaded!
- [x] Compress csv outputs!
- [x] Save SUMO stdout for debugging.
- [x] Provide version number in each results output, as well as a copy of the configurations.
- [x] Make the Hull model read the vType configuration!
- [x] Make the Hull and SUMO models modular, so that the user can choose which one to run. Currently, both are run.
- [x] Automatically run steps with default options.
- [x] ~~Seperate EV and BEV simulations?~~
- [x] Merge Chris Hull's BEV/EV simulation model.
- [x] Add MacOS instructions
    - [x] Install pigz with macports
    - [x] Export SUMO_HOME environment variable with `export SUMO_HOME="/opt/homebrew/share/sumo"` in `./.bash_profile` or `~/.zshrc`.
- [x] Debug T6001 output, of Chris Hull.
- [x] Automatically compress results with
      ```sh
      tar -c -I 'xz -9 -T10' -f 'Battery.out.csv.tar.xz' ./T*/*/Battery.out.csv
      ```
- [x] ~~Seperate common data-preprocessing tasks from `Kampala_UTX.py`.~~
- [x] Make the package pip-installable.
- [x] There are a collection of hacky scripts which are scattered throughout the source tree. They typically follow the namin scheme "dirty_*.py". These scripts are used by copying and pasting them into a certain folder in the scenario's direcory tree. That is not ideal for the software's usability. Therefore the hacky scripts need to be integrated into the main program, so that it is run automatically like any of the other submodules in this project (i.e. without manual copy-pasting).
