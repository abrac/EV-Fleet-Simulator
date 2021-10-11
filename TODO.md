<!-- Note: This is a markdown file. Use a markdown editor to easily edit and
     view this file. Just search the web for a nice markdown editor (like
     Ghostwriter). -->

- [ ] Automatically compress results with
      ```sh
      tar -c -I 'xz -9 -T10' -f 'Battery.out.csv.tar.xz' ./T*/*/Battery.out.csv
      ```
- [ ] Seperate common data-preprocessing tasks from `Kampala_UTX.py`.
- [ ] Make the package pip-installable.
- [x] There are a collection of hacky scripts which are scattered throughout the source tree. They typically follow the namin scheme "dirty_*.py". These scripts are used by copying and pasting them into a certain folder in the scenario's direcory tree. That is not ideal for the software's usability. Therefore the hacky scripts need to be integrated into the main program, so that it is run automatically like any of the other submodules in this project (i.e. without manual copy-pasting).
- [ ] Create a sub-module which does battery-sizing estimates.
