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

   It must be in the following format: 

   | Longitude | Latitude |
   |-----------|----------|
   | 18.6      | -34.3    |
   | 19.0      | -34.3    |
   | 19.0      | -33.7    |
   | 18.6      | -33.7    |

1. Download an `osm.pbf` file which represents the country. Currently, they are 
   available from [geofabrik.de](
   https://download.geofabrik.de/). 
   [Other sources](
   https://wiki.openstreetmap.org/wiki/Planet.osm#Country_and_area_extracts) 
   are available and [more information about pbf files](
   https://wiki.openstreetmap.org/wiki/PBF_Format) can be found at the OSM 
   wiki.

1. Copy the `.osm.pbf` file and `./Map_Construction/pbf_to_osm.sh` to
   `$Scenario_Dir/_Inputs/Map/Construction`. 

   Open the copy of the `.sh` file in a text editor, and modify the
   `-b=<min_lon>,<min_lat>,<max_lon>,<max_lat>` to correspond with the values
   added to `boundary.csv`. 

   > E.g. `-b=18.6,-34.3,19.0,-33.7`

   Run the modified `.sh` file to convert the `.osm.pbf` file to an `osm` file,
   while cropping to the specified boundary.

1. Copy `./Map_Contruction/net_convert.sh` to the `Construction` directory.
   Run the script to convert the `.osm` file to `.net.xml`. 

   Sometimes this step will throw many warnings due to badly formed data. If
   you have time, try and fix the warnings by editing the `.net.xml` file. (I
   almost always ignore the warnings though...)

1. Give permission for taxis to drive on almost any road.

    Open the generated `.net.xml` file in the `Netedit` software which comes
    with SUMO. 

    Change to "Selection" mode. (`Menu Bar > Edit > Select mode`)

    In the left side-panel,  make sure "Modification Mode" is "add", and that
    there *0 items* currently selected. Scroll down to the "Match Attribute"
    section. 

    Select "edge" in the first combo-box, and "type" in the second combo-box.
    In the text-box type each of the following, pressing "Apply selection" each
    time.

    <!--
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
    1. highway.track
    1. highway.services
    1. highway.service

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
    1. If the raw data is floating car data (FCD), i.e. GPS traces, proceed to 
       the next step.
    1. If the raw data is of the General Transit Feed Specification (GTFS),
       unzip the GTFS archive into: `$Scenario_Dir/_Inputs/Traces/Original/
       GTFS`.
    1. Make sure that GTFS data complies to the following caveats:
        1. Arrival times *and* departure times must be defined in 
           `stop_times.csv`. They should not be equal to the same value.
        1. `frequencies.txt` must be defined for each and every trip.
1. Create a script in `$Src_Dir/data_processing_ev/scenario_initialisation/
   Data_Pre-Processing` to transform the gps-traces to be in the format of the 
   original Stellenbosch data.
    1. Specifically, the data must be in CSV format with the below format.
    1. The headings in marked with `*` are required. 
    1. Headings marked with `+` are conditionally required.
        1. StopID only required for GTFS data.

   |          | GPSID | Time *                    | Latitude *  | Longitude * | Altitude  | Heading   | Satellites | HDOP[^1] | AgeOfReading | DistanceSinceReading | Velocity * | StopID + |
   |----------|-------|---------------------------|-------------|-------------|-----------|-----------|------------|----------|--------------|----------------------|------------|----------|
   | Datatype | str   | str                       | float       | float       | int/float | int/float | int        | float    | int          | int                  | int        | str      |
   | units    | -     | 'yyyy/mm/dd HH:MM:ss APM' | [-]11.11111 | [-]11.11111 | meters    | degrees   | -          | meters   | minutes?     | meters               | km/h       | -        |

1. Copy the script to the `$Scenario_Dir/_Inputs/Traces/` directory and run it.
1. Make sure the processed traces are in `$Scenario_Dir/_Inputs/Traces/
   Processed`.

[^1]: Horizontal Dilution of Precision. Lower is better.
