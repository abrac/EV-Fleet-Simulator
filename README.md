Installation
============

1. Clone the project to your local PC, into a folder of your choice. 
    ```sh
    mkdir -p ~/Applications  # Make a folder called "Applications" in your home directory.
    cd ~/Applications  # Change your terminal's directory to the folder you created.
    git clone https://gitlab.com/eputs/embt-sim.git  # Clone the git repository into the new folder.
    ```
2. Open the *src* folder, and make a symbolic-link (a.k.a "shortcut") of `main.py` in `~/.local/bin/`.
    ```sh
    cd ./embt-sim/src/
    ln -s main.py ~/.local/bin/embt-analysis
    ```
3. The program is now installed. You can simply enter `embt-analysis` in your terminal to run the software.

Usage
=====

Here are the steps to create your first simulation scenario:

1. Create a folder for the scenario that you want to create.
    ```sh
    mkdir -p ~/Documents/minibus-taxis-cape-town/
    ```
2. Run embt-analysis, specifying the the scenario directory.
    ```sh
    embt-analysis ~/Documents/minibus-taxis-cape-town/
    ```
3. Follow the prompts. It will ask you which "step" you want to execute. Since this is a new scenario, you want to run step 0, which initialises the scenario's directory structure. Enter `0` to initialise the scenario.

4. After you've initialised the scenario, follow the steps in [initialisation-instructions.md](./src/data_processing_ev/scenario_initialisation/initialisation-instructions.md). This will give you instructions on how to import a map of the area of your scenario, and how to import GPS traces of the vehicle that you are working with.

5. Proceed to run the remaining steps of the program, by running:
    ```sh
    embt-analysis ~/Documents/minibus-taxis-cape-town/
    ```
    again. Iteratively selecting the next steps in the list of steps.

6. The list of steps are (currently):
    ```
      0. scenario_initialisation
      1. data_visualisation
          1. mapping
          2. route_animation  <-- (Deprecated! Please don't run.)
      2. spatial clustering and filtering
          1. spatial_clustering
          2. date_filtering_and_separation
      3. temporal_clustering (and filtering)
      4. routing
      5. simulation
      6. results_generation
    ```
---

Additional usage instructions can be found by entering 
```sh 
embt-analysis --help
```
in a terminal.

Contributing
============

Making changes to the program is relatively easy. 

Let's make the proram greet us with "Hello world!" when we run `embt-analysis`.

1. Open the src folder in your cloned repository.
    ```sh
    cd ~/Applications/embt-sim/src/
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
5. Save the file, open a terminal, and run embt-analysis. It should say "Hello world!" before continuing with the program.

You will see that main.py consists of a function called `run`. This function is responsible for running the *sub-module* corresponding to the step.

If you want to edit the functionality of a particular step, you have to open the module corresponding to that step. Each module is contained in a folder in the `src/data_processing_ev` folder. Simply open the folder of that module and edit its python scripts.

If you have made any changes that are worth contributing back to the main project, we would be glad to accept them. Please do the following:

1. Open the repository on GitLab and create a *fork*.

2. Clone your fork to your PC. 

3. Make the changes to your fork.

4. Make a merge-request on GitLab, to merge your changes into our repository.

---

For more help, please don't hesitate to contact me at `chris <abraham-without-the-A's> [at] gmail [dot] com`
