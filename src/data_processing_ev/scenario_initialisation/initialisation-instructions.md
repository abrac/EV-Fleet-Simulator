Instructions
============

> Definitions: 
> 
> `$Scenario_Dir/` refers to the root directory of the scenario.
> 
> `$Src_Dir/` refers to the "src" directory in the EV-Fleet-Sim repository.

1. Run the scenario initialisation to create the folder structure in your 
   scenario directory.

1. Go to `$Scenario_Dir/_Inputs/Map/Boundary` and create `boundary.csv`.

   It must be a csv file with the following format: 

   ```
   +-----------+-----------+
   | Longitude | Latitude  |
   |-----------|-----------|
   | <min_lon> | <min_lat> |
   | <max_lon> | <min_lat> |
   | <max_lon> | <max_lat> |
   | <min_lon> | <max_lat> |
   +-----------+-----------+
   ```

1. Download an `osm.pbf` file which represents the country. Currently, they are 
   available from [geofabrik.de](
   https://download.geofabrik.de/). 
   [Other sources](
   https://wiki.openstreetmap.org/wiki/Planet.osm#Country_and_area_extracts) 
   are available and [more information about pbf files](
   https://wiki.openstreetmap.org/wiki/PBF_Format) can be found at the OSM 
   wiki.

1. Copy the `.osm.pbf` file to `$Scenario_Dir/_Inputs/Map/Construction`. 

   You will find a bash script called `pbf_to_osm.sh` in the `Construction`
   directory. Open it in a text editor, and modify the
   `-b=<min_lon>,<min_lat>,<max_lon>,<max_lat>` to correspond with the values
   added to `boundary.csv`. 

   > E.g. `-b=18.6,-34.3,19.0,-33.7`

   Run the modified `.sh` file to convert the `.osm.pbf` file to a `.osm` file,
   while cropping to the specified boundary.

1. You will also find a `net_convert.sh` file in the `Construction` directory.
   Run the script to convert the `.osm` file to `.net.xml` (the file-format 
   compatible with SUMO).

   Sometimes this step will throw many warnings due to badly formed data. If
   you have time, try and fix the warnings by editing the `.net.xml` file. (I
   almost always ignore the warnings though...)

1. Give permission for taxis (or whatever [vehicle type](
   https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html#vehicle_types
   ) you want to use) to drive on almost any road.

   Open the generated `.net.xml` file in the `Netedit` software which comes
   with SUMO. 

   Change to "Selection" mode. (`Menu Bar > Edit > Select mode`)

   In the left side-panel,  make sure "Modification Mode" is "add", and that
   there *0 items* currently selected. Scroll down to the "Match Attribute"
   section. 

   Select "edge" in the first combo-box, and "type" in the second combo-box.
   In the text-box type each of the following, pressing "Apply selection" each
   time.

   1. highway.track
   1. highway.services
   1. highway.service

   <!-- <COMMENT> LIST OF EDGE TYPES:
   1. highway.motorway
   1. highway.motorway_link
   1. highway.trunk
   1. highway.trunk_link
   1. highway.primary
   1. highway.primary_link
   1. highway.secondary
   1. highway.secondary_link *
   1. highway.tertiary
   1. highway.tertiary_link
   1. highway.unclassified
   1. highway.residential *
   1. highway.living_street
   1. highway.service
   1. highway.services
   1. highway.track
   -->

   Note that each time you press the "Apply selection button", *Edges* (i.e. 
   roads) on the map are being highlighted in blue. This indicates that the
   edges are selected.

   Switch to "Inspect" mode. (`Menu Bar > Edit > Inspect mode`)

   Click on any of the selected edges to inspect the entire selection.  Note:
   Netedit may hang if your selection is very large. If you are unable to
   proceed, try to repeat these steps with a one edge type at a time.

   Append "passenger" and "taxi" to the "allowed" field.

   Save the `.net.xml` file.

1. Move the resulting `net.xml` file in `$Scenario_Dir/_Inputs/
   Map`

1. Copy raw vehicle data to `$Scenario_Dir/_Inputs/Traces/Original`.

    If the raw data is floating car data (FCD), i.e. GPS traces, proceed to the 
    next step.

    If the raw data is of the General Transit Feed Specification (GTFS), unzip
    the GTFS archive into: `$Scenario_Dir/_Inputs/Traces/Original/ GTFS`.

    Make sure that GTFS data complies to the following caveats:

    1. Arrival times *and* departure times must be defined in `stop_times.csv`.
       They should not be equal to the same value.

    1. `frequencies.txt` must be defined for each and every trip.

1. If your input data is *floating car data*, you will need to create a script to 
   transform the gps-traces to a CSV file in the format that is compatible with 
   EV-Fleet-Sim. Please see the table below which outlines the columns that 
   are required in the CSV file, and the format that their values need to 
   conform to.

   There are some template scripts in `$Src_Dir/data_processing_ev/
   scenario_initialisation/Data_Pre-Processing/` to help you create your 
   script.

   However, if your input data is of the *GTFS* data format, you can use the 
   `GTFS_Convert.r` and `GTFS_Splitter.py` scripts which are in the 
   `Data_Pre-Processing` folder. You can use them as-is. No changes necessary.

   ```
   +----------+-------+---------------------------+-------------+--
   |          | GPSID | Time *                    | Latitude *  |   ...
   |----------|-------|---------------------------|-------------|--
   | Datatype | str   | str                       | float       |   ...
   | units    | -     | 'yyyy/mm/dd HH:MM:ss APM' | [-]11.11111 |
   +----------+-------+---------------------------+-------------+--

        -------------+-----------+-----------+------------+----------+--
    ...  Longitude * | Altitude  | Heading   | Satellites | HDOP[^1] |   ...
        -------------|-----------|-----------|------------|----------|--
    ...  float       | int/float | int/float | int        | float    |   ...
         [-]11.11111 | meters    | degrees   | -          | meters   |
        -------------+-----------+-----------+------------+----------+--

        --------------+----------------------+------------+--------------+
    ...  AgeOfReading | DistanceSinceReading | Velocity * | StopID[^2] + |
        --------------|----------------------|------------|--------------|
    ...  int          | int                  | int        | str          |
         minutes?     | meters               | km/h       | -            |
        --------------+----------------------+------------+--------------+

   Notes:
   ------

   The headings marked with `*` are required. 
   Headings marked with `+` are conditionally required.

   [^1]: Horizontal Dilution of Precision. Lower is better.
   [^2]: StopID column only required for GTFS input data.
   ```

1. Copy the scripts to the `$Scenario_Dir/_Inputs/Traces/` directory and run 
   them.

1. Make sure the processed traces are in `$Scenario_Dir/_Inputs/Traces/
   Processed`.
