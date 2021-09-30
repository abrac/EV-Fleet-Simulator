Instructions
============

1. Run the scenario initialisation to create the folder structure in your 
   scenario directory.
2. Go to `$Scenario_Dir/_Inputs/Map/Boundary` and create `boundary.csv`.
    1. It must be in the following format: 

| Longitude | Latitude   |
|-----------|------------|
| 18.656884 | -34.229224 |
| 18.969438 | -34.229224 |
| 18.969438 | -33.786222 |
| 18.656884 | -33.786222 |

3. Download an `osm.pbf` file which represents the country. Currently, they are 
   available from [geofabrik.de](
   https://download.geofabrik.de/). 
   [Other sources](
   https://wiki.openstreetmap.org/wiki/Planet.osm#Country_and_area_extracts) 
   are available and [more information about pbf files](
   https://wiki.openstreetmap.org/wiki/PBF_Format) can be found at the OSM 
   wiki.
3. Run `./Map_Construction/pbf_to_osm.sh` to convert `<country>.osm.pbf` to 
   osm, while cropping to the specified boundary. Remember to change the 
   boundary by editing the script.
4. Run SUMO's `netconvert` (or SAGA's `scenarioFromOSM.py`) to convert the osm 
   to `.net.xml`.
    1. See `./Map_Contruction/net_convert.sh`.
5. Open the `.net.xml` file in Netconvert, highlight `highway.service` edges, 
   and add `pedestrian` and `taxi` to allowed vehicle types.
5. Give permission for taxis to drive on almost any road.
    1. Open the generated `.net.xml` file in `netconvert`. 
    1. Change to "Selection" mode. In the left side-panel.
    1. Make sure "Modification Mode" is "add", and that there is nothing
       currently selected.
    1. Scroll down to the "Match Attribute" section. 
    1. Select "edge" in the first combo-box, and "type" in the second 
       combo-box. In the text-box type each of the following, pressing "Apply
       selection" each time.
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
        2. highway.services
        3. highway.service
    1. Switch to "Inspect" mode.
    1. Click on any of the selected edges to inspect the entire selection. 
       Note: Netedit may hang if your selection is very large. If you are
       unable to proceed, try to repeat these steps with a one edge type at a time.
    1. Append "passenger" and "taxi" to the "allowed" field.
    1. Save the `.net.xml` file.
6. Move the resulting `net.xml` file in `$Scenario_Dir/_Inputs/
   Map`
6. Copy raw vehicle data to `$Scenario_Dir/_Inputs/Traces/Original`.
    1. If the raw data is floating car data (FCD), i.e. GPS traces, proceed to 
       the next step.
    2. If the raw data is of the General Transit Feed Specification (GTFS),
       unzip the GTFS archive into: `$Scenario_Dir/_Inputs/Traces/Original/
       GTFS`.
    3. Make sure that GTFS data complies to the following caveats:
        1. Arrival times *and* departure times must be defined in 
           `stop_times.csv`. They should not be equal to the same value.
           (However, I am working on a work-around so that the software
           accommodates such data).
7. Create a script in `$Src_Dir/data_processing_ev/scenario_initialisation/
   Data_Pre-Processing` to transform the gps-traces to be in the format of the 
   original Stellenbosch data.
    1. Specifically, the data must be in CSV format with the below format.
    2. The headings in marked with `*` are required. 
    3. Headings marked with `+` are conditionally required.
        1. StopID only required for GTFS data.

|          | GPSID | Time *                    | Latitude *  | Longitude * | Altitude  | Heading   | Satellites | HDOP[^1] | AgeOfReading | DistanceSinceReading | Velocity * | StopID + |
|----------|-------|---------------------------|-------------|-------------|-----------|-----------|------------|----------|--------------|----------------------|------------|----------|
| Datatype | str   | str                       | float       | float       | int/float | int/float | int        | float    | int          | int                  | int        | str      |
| units    | -     | 'yyyy/mm/dd HH:MM:ss APM' | [-]11.11111 | [-]11.11111 | meters    | degrees   | -          | meters   | minutes?     | meters               | km/h       | -        |

7. Copy the script to the `$Scenario_Dir/_Inputs/Traces/` directory and run it.
8. Make sure the processed traces are in `$Scenario_Dir/_Inputs/Traces/
   Processed`.

[^1]: Horizontal Dilution of Precision. Lower is better.
