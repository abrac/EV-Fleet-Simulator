<!-- Note: This is a markdown file. Use a markdown editor to easily edit and
     view this file. Just search the web for a nice markdown editor (like
     Ghostwriter). -->

> Notice: Please visit https://gitlab.com/eputs/ev-fleet-sim to ensure that you
> are viewing the official, up-to-date version of this repository.

Electric-Vehicle Fleet Simulator
================================

This program is used to predict the energy usage of a fleet of electric
vehicles. The program receives as input GPS traces of each of the vehicles of
the fleet. These GPS traces can be obtained, for example, by installing
tracking devices onto the vehicles of a fleet for which you want to predict the
electrical energy usage. This is especially useful for projects whereby an
existing petrol/diesel fleet is to be converted to electric vehicles. The
program will analyse the vehicle's driving and stopping patterns in order to
predict the amount of energy used and the amount of time that the vehicle can
be charged during the average day. In addition, the program makes provisions to
calculate how much of the energy can be provided for by renewable-energy
sources.

Please refer to the accompanying open-access journal article publication: [Ray
of hope for sub-Saharan Africa's paratransit: Solar charging of urban electric
minibus taxis in South Africa](https://doi.org/10.1016/j.esd.2021.08.003). The
article shows how this program can be used to derive meaningful results.

Licensing
=========

This software is [licensed under GPLv3](./LICENSE)

If you use the software, or a derivative thereof, you are required to
attribute the original authors using the following citation:

> Abraham, C. J., Rix, A. J., Ndibatya, I., & Booysen, M. J. (2021). Ray of
> hope for sub-Saharan Africa's paratransit: Solar charging of urban electric
> minibus taxis in South Africa. Energy for Sustainable Development, 64,
> 118-127. https://doi.org/10.1016/j.esd.2021.08.003

<details><summary>Bibtex</summary>

```
@article{abraham2021,
title = {Ray of hope for sub-Saharan Africa's paratransit: Solar charging of urban electric minibus taxis in South Africa},
journal = {Energy for Sustainable Development},
volume = {64},
pages = {118-127},
year = {2021},
issn = {0973-0826},
doi = {https://doi.org/10.1016/j.esd.2021.08.003},
url = {https://www.sciencedirect.com/science/article/pii/S0973082621000946},
author = {C.J. Abraham and A.J. Rix and I. Ndibatya and M.J. Booysen},
keywords = {Electric vehicle, Paratransit, Minibus taxi, Demand management, Renewable energy},
abstract = {Minibus taxi public transport is a seemingly chaotic phenomenon in the developing cities of the Global South with unique mobility and operational characteristics. Eventually this ubiquitous fleet of minibus taxis is expected to transition to electric vehicles, which will result in an additional energy burden on Africa's already fragile electrical grids. This paper examines the electrical energy demands of this possible evolution, and presents a generic simulation environment to assess the grid impact and charging opportunities. We used GPS tracking and spatio-temporal data to assess the energy requirements of nine electric minibus taxis as well as the informal and formal stops at which the taxis can recharge. Given the region's abundant sunshine, we modelled a grid-connected solar photovoltaic charging system to determine how effectively PV may be used to offset the additional burden on the electrical grid. The mean energy demand of the taxis was 213kWh/d, resulting in an average efficiency of 0.93kWh/km. The stopping time across taxis, a proxy for charging opportunity, ranged from 7.7 h/d to 10.6 h/d. The energy supplied per surface area of PV to offset the charging load of a taxi while stopping, ranged from 0.38 to 0.90kWh/m2 per day. Our simulator, which is publicly available, and the results will allow traffic planners and grid operators to assess and plan for looming electric vehicle roll-outs.}
}
```

</details>


Installation
============

[![PyPI Version](https://img.shields.io/pypi/v/ev-fleet-sim)](https://pypi.org/project/ev-fleet-sim/)

1. Make sure that you have installed the dependencies listed in [depencies.md](
   ./dependencies.md)

1. Clone the project to your local PC, into a folder of your choice. 

   ```sh
   # Note: Commands shown in this file work in Bash.
   mkdir -p ~/Applications  # Make a folder called "Applications" in your home directory.
   cd ~/Applications  # Change your terminal's directory to the folder you created.
   git clone https://gitlab.com/eputs/ev-fleet-sim.git  # Clone the git repository into the new folder.
   ```

1. To run the software, open a terminal in the *src* folder of the repository, 
   and enter `./main.py` to run the program. However, if you are on Windows, 
   you will need to enter `python main.py` instead.

1. Optional: Open the *src* folder, and make a symbolic-link (a.k.a "shortcut")
   of `main.py` to `~/.local/bin/ev-fleet-sim` (If running Linux. Not sure how
   to do this in Windows & Mac). This will allow you to run the
   software *anywhere* by entering `ev-fleet-sim` in your terminal.

   ```sh
   cd ./ev-fleet-sim/src/  # Change to the src directory.
   ln -s main.py ~/.local/bin/ev-fleet-sim  # Create a symbolic-link of main.py.
   cd ~/Documents/minibus-taxis-cape-town  # Change to some directory where main.py is not present.
   ev-fleet-sim  # Run the program from there.
   ```

Usage
=====

Here are the steps to create your first simulation scenario:

1. Create a folder for the scenario that you want to create.

    ```sh
    mkdir -p ~/Documents/minibus-taxis-cape-town/
    ```

2. Open a terminal in the *src* directory, and run `main.py`, optionally
   specifying the scenario directory.

    ```sh
    ev-fleet-sim ~/Documents/minibus-taxis-cape-town/
    ```

3. Follow the prompts. It will ask you which "step" you want to execute. Since
   this is a new scenario, you want to run step 0, which initialises the
   scenario's directory structure. Enter `0` to initialise the scenario.

4. After you've initialised the scenario, follow the steps in
   [initialisation-instructions.md](
   ./src/data_processing_ev/scenario_initialisation/initialisation-instructions.md).
   This file will give you instructions on how to import a map of the area of
   your scenario, and how to import GPS traces of the vehicle that you are
   working with.

5. Proceed to run the remaining steps of the program, by running again:

    ```sh
    ev-fleet-sim ~/Documents/minibus-taxis-cape-town/
    ```

    Run the script iteratively, each time selecting the next steps in the list 
    of steps.

Notes: 

* The list of steps will be presented to you when your run `ev-fleet-sim`.  You
  can also view the list of steps by opening the file
  `./src/data_processing_ev/ __init__.py`, and finding the lines starting with
  `MODULES`.

* In the same file, if you look for the line starting with
  `SCENARIO_DIR_STRUCTURE`, there is a *dictionary* which specifies the
  directory structure that will be created in the scenario. The order of
  directories shown in this dictionary is approximately the order of
  directories to which the program generates its outputs. I.e. Step *1* will
  generate its outputs in the *1st* directory specified in the dictionary.

* Additional usage instructions can be found by entering 

  ```sh 
  ev-fleet-sim --help
  ```

  in a terminal.

Contributing
============

Now I'll show you how to make changes to the program, and how to upload your
changes so that we can all benefit from them.

Let's make the program greet us with "Hello world!" when we run `main.py`.

1. Open the src folder in your cloned repository.

   ```sh
   cd ~/Applications/ev-fleet-sim/src/
   ```

2. Open the `main.py` file in your favourite text-editor.

   ```sh
   gedit main.py
   ```

3. Go down to the line which says:

    ```python
    if __name__ == "__main__":
    ```

4. If that is line 75, insert in line 76:

    ```python
    print("Hello world!")
    ```

5. Save the file, open a terminal, and run `main.py`. It should say "Hello
   world!" before continuing with the program.

You will see that main.py consists of a function called `run`. This function is
responsible for running the *sub-module* corresponding to the step.

If you want to edit the functionality of a particular step, you have to open
the module corresponding to that step. Each module is contained in a folder in
the `./src/data_processing_ev` folder. Simply open the folder of that module
and edit its python scripts.

If you have made any changes that are worth contributing back to the main
project, we would gratefully accept them. This is how you do it:

1. Open this repository in GitLab, and press the *Fork* button.

2. Clone your fork to your PC and make the changes that you want to make on
   your fork.

   ```sh
   cd ~/Applications
   git clone https://gitlab.com/eputs/ev-fleet-sim-FORK.git
   # Make some changes...
   ```

3. *Commit* the changes to the local copy of your fork. 

   ```sh
   git add file-i-changed
   git add another-file-i-changed
   git commit -m a-short-summary-of-the-change
   ```

3. Push the commits to the online copy of your fork.

   ```sh
   git push
   ```

4. Open your fork's repository in GitLab and create a *pull-request*. This will
   notify me that you would like to send some changes (commits) to me. With a
   few clicks I will be able to accept the changes that I like, and reject the
   changes that I don't.

Why don't you create a fork, do this Hello-World exercise on your fork, and
create a pull-request. Of course, I will reject the pull-request, but consider
it the "initiation rites" of EV-Fleet-Sim 😉.


Getting Support
---------------

Welcome to our EV-Fleet-Sim community! You can join our community's Matrix
channel (an open-source alternative of Microsoft Teams):
https://matrix.to/#/#ev-fleet-sim:matrix.org.

For more help, please don't hesitate to contact me on my e-mail address: 
`chris <abraham-without-the-A's> [at] gmail [dot] com` or via Matrix:
https://matrix.to/#/@abrac:matrix.org.

Also remember to press the "star" and "notification bell" on the top of the
GitLab page. That way, you will be notified of the latest commits and
pull-requests.
