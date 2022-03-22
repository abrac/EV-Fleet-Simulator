---
title: Usage
---

Here are the steps to create your first simulation scenario:

EV-Fleet-Sim requires you to create a folder for the scenario that you would like to simulate. This folder will contain the input data, and the simulation results.

1. Create a folder for the scenario that you want to create. Let's call this folder `<simulation-dir>`.

2. Open a terminal and run `ev-fleet-sim`. The program will ask you for the scenario directory. Enter in `<simulation-dir>`. Tip, you can also run `ev-fleet-sim <simulation-dir>` to skip the prompt.

3. Follow the prompts. The program will ask you which "step" you want to execute. Since
   this is a new scenario, you want to run step 0, which initialises the
   scenario's directory structure. Enter `0` to initialise the scenario.

4. After you've initialised the scenario, a file called `initialisation-instructions.md` will appear in `<simulation-dir>`. Follow the steps in this file. The instructions can also be found here: [initialisation-instructions.md](https://gitlab.com/eputs/ev-fleet-sim/-/blob/master/src/data_processing_ev/scenario_initialisation/initialisation-instructions.md).

   This file will give you instructions on how to import a map of the area of your scenario, how to import GPS traces of the vehicle that you are working with, and how to import whether files for the renewable energy-simulation. The instructions are quite details, so please take your time at this step and [contact us](/contact.html) if you face any difficulties!

5. Proceed to run the remaining steps of the program, by running again:

    ```sh
    ev-fleet-sim <simulation-dir>
    ```

    Run the script iteratively, each time selecting the next steps in the list 
    of steps.

Notes: 

* The list of steps will be presented to you when your run `ev-fleet-sim`.

* After step 0, you will find a `readme.json` file in `<simulation-dir>`. This file contains a list with the directory structure of the scenario. The order of sub-directories shown in this list is the order in which the program generates its outputs. I.e. Step *1* will generate its outputs in the *1st* directory specified in the list.

* Additional usage instructions can be found by entering 

  ```sh 
  ev-fleet-sim --help
  ```

  in a terminal.
