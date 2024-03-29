---
title: Usage
---

Overview
========

Here are the steps to create your first simulation scenario:

EV-Fleet-Sim requires you to create a folder for the scenario that you would like to simulate. This folder will contain the input data, and the simulation results.

1. Create a folder for the scenario that you want to create. Let's call this folder `<scenario-dir>`.

2. Open a terminal and run `ev-fleet-sim`. The program will ask you for the scenario directory. Enter in `<scenario-dir>`. Tip, you can also run `ev-fleet-sim <scenario-dir>` to skip the prompt.

3. Follow the prompts. The program will ask you which "step" you want to execute. Since
   this is a new scenario, you want to run step 0, which initialises the
   scenario's directory structure. Enter `0` to initialise the scenario.

4. After you have done that, you will need to follow the steps in the [below initialisation instructions](#initialisation-instructions). The instructions are quite detailed, so please take your time at this step and [contact us]({{site.baseurl}}/contact.html) if you face any difficulties!

5. After you have initialised the scenario, proceed to run the remaining steps of the program, by running again:

    ```sh
    ev-fleet-sim <scenario-dir>
    ```

    Run the script with the steps that you would like to run. Typically, you will need to run all of the steps, but it depends on your needs.

    For example, if you open a terminal in `<scenario-dir>`, you can run: `ev-fleet-sim .`. This will return the following result:

        Available steps: 
        
        0: scenario_initialisation
        1: data_visualisation
            1.1: mapping
            1.2: route_animation
            1.3: map_size_calculation
        2: spatial_analysis
            2.1: spatial_clustering
            2.2: date_filtering_and_separation
            2.3: save_dates_remaining
        3: temporal_analysis
            3.1: stop_extraction
            3.2: stop_duration_box_plots
            3.3: temporal_clustering
        4.x: mobility_simulation
            4.1: routing  **OR** 4.2: fcd_conversion
        5.x: ev_simulation
            5.1: sumo_ev_simulation **OR** 5.2: hull_ev_simulation
        6: results_analysis
            6.1: ev_results_analysis
            6.2.x: reg_results_analysis
                6.2.1: pv_results_analysis **OR** 6.2.2: wind_results_analysis

        Specify steps to be run as a comma-seperated list of floats without spaces (e.g. '1,2.2,4'):       

    If you look carefully, the last line asks us to specify the steps that we want to run. Typically, we will want to run all the steps. Since we already ran step 0, you can enter `1,2,3,4.1,5.1,6.1,6.2` to run the remaining steps. 

    This will run the simulation, starting from step 1, and ending at step 6.2.

    Some of the steps provide two options. For example, for step 4, you should either run `4.1` or `4.2`. If you simply enter `4`, then the first option (i.e. `4.1`) will be selected.

Notes: 

* The list of steps are presented to you whenever your run `ev-fleet-sim`. If you already know what steps you want to run, you can skip the prompt by running: `ev-fleet-sim <scenario-dir> --steps <steps>`, where `<steps>` are the list of steps that you want to run.

* After step 0, you will find a `readme.json` file in `<scenario-dir>`. This file contains a list with the directory structure of the scenario. The order of sub-directories shown in this list is the order in which the program generates its outputs. I.e. Step *1* will generate its outputs in the *1st* directory specified in the list.

* Additional usage instructions can be found by entering `ev-fleet-sim --help ` in a terminal.

* When using Windows, always run `ev-fleet-sim` and all commands in this tutorial using `GIT Bash` instead of `command prompt` or `powershell`. You can always open `GIT Bash` from the start menu or by right clicking in a file explorer and selecting `GIT Bash Here`.

* You can quit the program at any time by pressing `ctrl + C`. 


Initialisation Instructions
===========================

> Definitions: 
> 
> `<scenario-dir>/` refers to the root directory of the scenario.

Firstly, run the scenario-initialisation step of EV-Fleet-Sim to create the
folder structure in your scenario directory (`<scenario-dir>`).

Initialising Trace Data
-----------------------

1. Copy your fleet's raw vehicle data to 
   `<scenario-dir>/_Inputs/Traces/Original`.

   EV-Fleet-Sim supports two data formats: [floating car data (FCD)](
   https://en.wikipedia.org/wiki/Floating_car_data) (also commonly referred to 
   as "GPS traces"), and [General Transit Feed Specification (GTFS)](
   https://gtfs.org/). (GTFS is a way of digitally representing 
   public-transport schedules.)

   If the raw data is FCD, proceed to the next step.

   If the raw data is GTFS data, it should be a zipped archive. Rename the
   archive to "GTFS_Orig.zip". Unzip the GTFS archive into: 
   `<scenario-dir>/_Inputs/Traces/Original/GTFS`.

   Make sure that GTFS data complies to the following caveats:

   1. Arrival times *and* departure times must be defined in `stop_times.csv`.
      They should not be equal to the same value.

   1. `frequencies.txt` must be defined for each and every trip.

1. As mentioned previously, your input data may be FCD or GTFS data. We need to
   convert the input data fromat to *CSV files* that can be read by
   EV-Fleet-Sim. Please see the [table
   below](#table%3A-csv-input-format-for-ev-fleet-sim) which outlines the
   columns that are required in each CSV file, and the format that their values
   need to conform to.

   ##### FCD Conversion

   If your input data is floating car data, you will need to create a
   script to transform the gps-traces to CSV files. You must generate one CSV 
   file per vehicle in the fleet.

   Some template scripts are available on our [Git repository](https://gitlab.com/eputs/ev-fleet-sim/-/tree/master/src/data_processing_ev/scenario_initialisation/Data_Pre-Processing) to help you create your script.

   ##### GTFS Conversion

   If your input data is of the GTFS data format, you should use the `GTFS_Convert.r` and `GTFS_Splitter.py` scripts which are also in our [Git repository](https://gitlab.com/eputs/ev-fleet-sim/-/tree/master/src/data_processing_ev/scenario_initialisation/Data_Pre-Processing). You can use them as-is. No changes should be necessary. The scripts will generate one csv file per [trip](https://gtfs.org/reference/static#dataset-files) defined in the GTFS data.

   First, copy these two scripts to the `<scenario-dir>/_Inputs/Traces/` directory.

   Secondly, run `GTFS_Convert.r`, as follows: 

   Open a terminal in the `Traces` directly. Run the `R` command, to open an R prompt. Enter `source(GTFS_Convert.r)` into the prompt. Follow the various prompts until the script returns you to the R prompt. Exit the R prompt, by entering `quit()` and entering `n`.

   After this, extract the new file:
   `<scenario-dir>/_Inputs/Traces/Original/GTFS.zip` to 
   `<scenario-dir>/_Inputs/Traces/Original/GTFS/`.

   Finally, run the `GTFS_Splitter.py` script. It will tell you the maximum
   and minimum GPS coorindates encountered. Hold onto these values. They will
   be useful when [generating the road-network](#initialising-road-network). Press enter to continue, and waith for the script to complete. The data files compatible with EV-Fleet-Sim will be found in the `Processed` directory.

   ##### Table: CSV input format for EV-Fleet-Sim

   |          | GPSID | Time *                       | Latitude *  |   Longitude * | Altitude[^4] + | Heading   | Satellites | HDOP[^1] |  AgeOfReading | DistanceSinceReading | Velocity * | StopID[^2] + |
   |----------|-------|------------------------------|-------------|---------------|----------------|-----------|------------|----------|---------------|----------------------|------------|--------------|
   | Datatype | str   | str                          | float       |   float       | int/float      | int/float | int        | float    |  int          | int                  | int        | str          |
   | units    | -     | 'yyyy-mm-dd hr24:MM:ss' [^3] | [-]11.11111 |   [-]11.11111 | meters         | degrees   | -          | meters   |  minutes?     | meters               | km/h       | -            |


   > Note:
   >
   > The headings marked with `*` are required.
   >
   > Headings marked with `+` are conditionally required.
   >
   > Unmarked headings are not required, and are not currently used by EV-Fleet-Sim. They may be used in the future.
   > 
   > If you are not using a coloumn, leave its fields blank. (i.e. Don't fill it with zeroes.)

1. Copy the script(s) to the `<scenario-dir>/_Inputs/Traces/` directory and run 
   them.

1. Make sure the processed traces are in 
   `<scenario-dir>/_Inputs/Traces/Processed`.

Vehicle Definition
------------------

1. Open the `<scenario-dir>/_Inputs/Configs/ev_template.xml` file in your
   favourite text editor.

1. Choose a [vehicle class](
   https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html#abstract_vehicle_class
   ) that will represent your fleet in SUMO. Try to choose the class that most
   closely represents the type of vehicle you are trying to simulate. If you 
   don't know which one to choose, choose the "passenger" vehicle type.

   (List of available vehicle classes: [SUMO Documentation](
    https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html#abstract_vehicle_class))  
   (Default parameters of the various vehicle classes: [SUMO Documentation](
    https://sumo.dlr.de/docs/Vehicle_Type_Parameter_Defaults.html))

1. In this file, find the `vType` *element*: 
   ```xml
   <vType ... >

       ⋮

   </vType>
   ```

   In the `vType` element, find the `vClass` *attribute*:
   ```xml
   <vType ... vClass="value" ... >
   ```

   Change this attribute's value to the desired vehicle class. Change the other
   attributes of the `vType` element if desired. (To read more about the
   attributes and what their values mean, refer: [SUMO
   documentation](
   https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html#available_vtype_attributes).)

   Find the `param` elements:
   ```xml
   <param  key="..."  value="..." >
   ```

   If desired, you can change the values of the various parameters to modify
   the electric-vehicle model.

   (Reference to electric vehicle parameters: [SUMO Documentation](
    https://sumo.dlr.de/docs/Models/Electric.html#defining_electric_vehicles))

Initialising Road Network
-------------------------

1. Go to `<scenario-dir>/_Inputs/Map/Boundary`. You will find a file called
   `boundary.csv`. It is a csv file with two columns. Each row represents the
   coordinates of each of the four points in the boundary box that you would
   like to create.

   |   Longitude |    Latitude |
   |------------:|------------:|
   | `<min_lon>` | `<min_lat>` |
   | `<max_lon>` | `<min_lat>` |
   | `<max_lon>` | `<max_lat>` |
   | `<min_lon>` | `<max_lat>` |

   This bounding box will be used during the simulation. For FCD, if a vehicle
   leaves this bounding box on a particular day, that day's data will be
   discarded. For GTFS data, it is recommended to create a bounding box
   corresponding to the values given by `GTFS_Splitter.py` as [explained
   previously](#gtfs-conversion).

1. Download an `osm.pbf` file which represents the country. Currently, they are
   available from [geofabrik.de](https://download.geofabrik.de/). [Other
   sources](
   https://wiki.openstreetmap.org/wiki/Planet.osm#Country_and_area_extracts)
   are available and more information about pbf files can be found at the [OSM
   wiki](https://wiki.openstreetmap.org/wiki/PBF_Format).

1. Copy the `.osm.pbf` file to `<scenario-dir>/_Inputs/Map/Construction`. From here on, we will refer to this directory as `<construction-dir>`.

   You will find a bash script called `pbf_to_osm.sh` in `<construction-dir>`. Open it in a text editor. In MacOS and Linux, find the line which has `--bbox <min_lon>,<min_lat>,<max_lon>,<max_lat>`. On Windows, find the line which has `-b=<min_lon>,<min_lat>,<max_lon>,<max_lat>`. Modify the line to correspond with the values added to `boundary.csv`. 

   > E.g: `--bbox 18.6,-34.3,19.0,-33.7`<br>
   > Or in Windows: `-b=18.6,-34.3,19.0,-33.7`

   Run the modified `.sh` file to convert the `.osm.pbf` file to a `.osm` file, while cropping to the specified boundary. This should produce a file called `square_boundary.osm`.

1. <a id="importing-elevation-data"></a> ***Optional step: Importing elevation data*** 

    By default, OSM does not include elevation data. If you would like to use elevation data to create a more realistic simulation, then you will need to follow the following instructions:


    <details markdown='1' style="background:#EEEEEE;padding: 0.5em;"><summary>Click to view</summary><br>

    By default, OSM does not include elevation data. Elevation is supported by EV-Fleet-Sim to create more realistic simulations. In order to use that functionality, elevation data needs to be overlayed on the input OSM file. To do this, you need to install `osmosis`, and the `osmosis-srtm` plug-in.

    To install them, follow the steps below:

    1.  Download and unpack [Osmosis 0.45](https://bretth.dev.openstreetmap.org/osmosis-build/osmosis-0.45.zip) in a folder in your computer. We will refer to the folder's directory as `<osmosis-dir>`.

    2.  Download the pre-built [jar file](https://github.com/locked-fg/osmosis-srtm-plugin/files/8027597/srtmplugin-1.1.2.jar.zip) and place it in `<osmosis-dir>/lib/default`.

        **OR:**

        Clone and build the [osmosis-srtm plug-in](https://github.com/locked-fg/osmosis-srtm-plugin.git) repo. 


    3.  Create a text file called `osmosis-plugins.conf` in the `<osmosis-dir>/config/` directory and add this line to the file: 

        `de.locked.osmosis.srtmplugin.SrtmPlugin_loader`

    Downloading elevation data:

    1.  Create and sign into an account on [Earth Explorer](https://earthexplorer.usgs.gov/)

    2.  Play around on the map until the entire area you want to cover is
        seen.

    3.  Click on the "Use map" button and continue to data set

    4.  Check the *Use Data Set Prefilter* box and type: "NASA SRTM3
        SRTMGL3" in the *Data Set Search* bar. Make sure only the checkbox
        named "NASA SRTM3 SRTMGL3" is ticked

    5.  Click the "Results" button.

    6.  For each of the results, click "Download Options" and download the
        HGT file.

    7.  Unzip all of the downloads and place them together in the directory: `<construction-dir>/Elevation/`

    Overlaying elevation data on OSM:

    1.  As instructed in the previous step of the Initialising Road Network instructions, you should have a file called `square_boundary.osm` in `<construction-dir>`. Rename this file to `map-without-elevation.osm`.

    2.  Run the following line in you command prompt:

        Linux/MacOS:

        ```sh
        <osmosis-dir>/bin/osmosis --read-xml <construction-dir>/map-without-elevation.osm --write-srtm locDir=<construction-dir>/Elevation/ repExisting=true tagName=ele --write-xml <construction-dir>/map-with-elevation.osm
        ```
        
        Windows (GIT Bash):

        ```sh
        <osmosis-dir>/bin/osmosis.bat --read-xml <construction-dir>/map-without-elevation.osm --write-srtm locDir=<construction-dir>/Elevation/ repExisting=true tagName=ele --write-xml <construction-dir>/map-with-elevation.osm
        ```

    4.  Rename `map-with-elevation.osm` to `square_boundary.osm` so that it can be used in the next step in the Initialising Road Network instructions.

    </details><br>

1. You will also find a `net_convert.sh` file in the `Construction` directory. We will run this script to convert the `.osm` file to a `.net.xml` file (the road-network file-format that is compatible with SUMO). 

   But before we do that, you may want to take note of the `.typ.xml` file that is present in the `Construction` directory. This is a *SUMO edge-type file*.

   The `.typ.xml` file defines which vehicle classes are allowed to access the various road types. It can optionally be modified to suit the context of the scenario.

   For example, gravel roads (`highway.track`) are common in developing countries, and hence it would be expected that vehicles have access to those roads in the simulation (since tar roads are less common). In cities of developed countries, where gravel roads are less common, it may be appropriate to restrict vehicles from accessing them.

1. <a id="setting-road-access-permissions"></a>***Optional step: Setting Road Access Permissions***

   Remember that [previously](#vehicle-definition) we chose a vehicle class to represent our fleet? Now we can specify which road-types allow access for our fleet's vehicle class.

   <details markdown='1' style="background:#EEEEEE;padding: 0.5em;"><summary>Click to view</summary><br>
   Open the `.typ.xml` file in your favourite text editor. For each road-type listed, change the `disallow` and `allow` attributes to allow/disallow your vehicle class. You can also change the other attributes, such as the speed limits on the various road types.

   (Description of the various road types: [OpenStreetMap Wiki](https://wiki.openstreetmap.org/wiki/Map_features#Highway))

   (More information on SUMO edge-type files: [SUMO Documentation](https://sumo.dlr.de/docs/SUMO_edge_type_file.html))
   </details><br>

1. Now we can proceed to run the `net_convert.sh` script. 

    The script will prompt you to ask you if the scenario has left- or right-handed traffic. It will also ask you if you want to import elevation data from the OSM file. This option will only work if you [import elevation data](#importing-elevation-data) as explained previously.

   As the script runs, sometimes it
   will throw many warnings due to badly formed data. If you have time, try and
   fix the warnings by editing the `.net.xml` file in NETEDIT. (I usually
   ignore the errors, because most can be safely ignored.)

   There is one warning that requires special mention: If you get the following
   warning: `Warning: Discarding unusable type ...`, it means that netconvert
   encountered a road type which is not defined in the `typ.xml` file.
   In such cases, netconvert will ignore roads which have not been defined.
   (I.e. It will pretend as if they didn't exist.) If the road type is
   important to you, add it to the `.typ.xml` file, and [set it's access
   permissions](#setting-road-access-permissions).

   Once the scrpt is done running, the `.net.xml` file can be found in: 
   `<scenario-dir>/_Inputs/Map/`, and a log of the warnings will be saved as a
   text file in the `Construction` directory.

   (Description of netconvert warnings and the recommended actions: [SUMO
   Documentation](
   https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html#warnings_during_import))

1. We have now initialised the road network. However, if you want to make small
   changes, or view the generated road network graphically, you can open the 
   `.net.xml` file with the `Netedit` software which comes with SUMO. 

   (Netedit usage instructions: [SUMO Documentation](https://sumo.dlr.de/docs/Netedit/index.html))

Initialising Weather Data
-------------------------

Because EV-Fleet-Sim also does renewable energy calculations, you may also put
weather data files in the `<scenario-dir>/_Inputs/Weather` directory. If you are
not interested in doing renewable energy calculations, you may leave the
directory empty.

This data can be obtained by first installing [SAM](https://sam.nrel.gov/), and
using its user interface to download the data for the location that you need. This processs  is described in the section ["Running renewable energy simulations"](#running-renewable-energy-simulations).

In the case of solar energy data, SAM should have data for almost any location
around the world. In the case of wind data, they only provide data for
locations in the USA. In case they don't have data for the location that you
are interested in, or if you just want to use your own dataset, you will need
to re-format the data into a format that can be read by SAM. 


<details markdown='1' style="background:#EEEEEE;padding: 0.5em;"><summary><a id="using-your-own-data"></a> Using your own data:</summary><br>

In case you want to use your own data, the following document describes the file format that is required for SAM weather files.

- For version 2020.2.29 r1 (current version as of writing) see: 
  [this PDF](https://sam.nrel.gov/images/web_page_files/sam-help-2020-2-29-r2_weather_file_formats.pdf)
- For future versions see: 
  [this webpage](https://sam.nrel.gov/weather-data/weather-data-publications.html)

</details>


Running renewable energy simulations
------------------------------------

### PV Simulations 

1. Create a new project, choosing the "Photovoltaic/PVWatts/No Financial Model" option.
1. In the "Location and Resource" tab, download the Weather file using the "Download Weather Files" panel. More information about how to do so, can be found by pressing the "Help" button.
    1. After the file has downloaded copy it from the default download location (typically a folder called "SAM Downloaded Weather Files" in your user's home directory) to `<scenario-dir>/_Inputs/Weather`.
    1. Alternatively, use your own data, using the information in the panel titled ["Using your own data"](#using-your-own-data). Copy the data to the `<scenario-dir>/_Inputs/Weather` directory.
    1. If you use your own data, ensure that the weather file is loaded in the "Weather Data Information" panel.
1. In the "System Design" tab, input the model parameters of the desired PV plant.
1. Click "File/Save" in order to save the file to `<scenario-dir>/REG_Simulation/SAM_Scenario_File/`
1. Click the "Simulate" button in SAM.
1. In the results, click on the "Time Series" tab. Select "Plane of array irradiance (W/m2)" so that it displays on the graph. Right click on the graph, and click "Save data to CSV..." and save it to `<scenario-dir>/REG_Simulation/Results/`.
1. Proceed to run the PV simulation from the relevant step in `ev-fleet-sim`.

### Wind Simulations 

<!-- TODO Complete documentation -->

Follow a similar procedure to that described in the [PV Simulations](#pv-simulations) section, except using the "Wind/No Financial Model" option of SAM.

---

### Footnotes
[^1]: Horizontal Dilution of Precision. Lower is better.
[^2]: StopID column only required for GTFS input data.
[^3]: E.g: 2022-04-17 09:03:49
[^4]: Only required if you would like to account for elevation in the EV model. Press the following link to see the [steps to import elevation data](#importing-elevation-data).
