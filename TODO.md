<!-- Note: This is a markdown file. Use a markdown editor to easily edit and
     view this file. Just search the web for a nice markdown editor (like
     Ghostwriter). -->

- [ ] Merge [fcv-fleet-sim](https://gitlab.com/eputs/fcv-fleet-sim) FCV simulation model.
- [ ] Provide version number in each results output, as well as a copy of the configurations.
- [ ] Make `^C` cancel more elegantly. Currently it throws useless debugging info.
- [ ] Improve SAM Simulation documentation!
- [ ] Save SUMO stdout for debugging.
- [ ] Create a sub-module which does battery-sizing estimates.
- [ ] Automatically run steps with default options.
- [ ] Add stop-location functionality.
- [ ] Make the Hull simulation multithreaded!
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
