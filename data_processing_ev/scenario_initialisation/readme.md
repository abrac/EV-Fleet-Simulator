Instructions
============

1. Run the scenario initialisation to create the folder structure in your scenario directory.
2. Go to `$Scenario_Dir/_Inputs/Map/Boundary` and create `boundary.csv`.
    1. It must be in the following format: 

| Longitude | Latitude   |
|-----------|------------|
| 18.656884 | -34.229224 |
| 18.969438 | -34.229224 |
| 18.969438 | -33.786222 |
| 18.656884 | -33.786222 |

3. Download an `osm.pbf` file which represents the country. Currently, they are available from [geofabrik.de](https://www.geofabrik.de/data/download.html). [Other sources](https://wiki.openstreetmap.org/wiki/Planet.osm#Country_and_area_extracts) are available and [more information about pbf files](https://wiki.openstreetmap.org/wiki/PBF_Format) can be found at the OSM wiki.
3. Run `./Map_Construction/pbf_to_osm.sh` to convert `<country>.osm.pbf` to osm, while cropping to the specified boundary. Remember to change the boundary by editing the script.
4. Run SAGA's `scenarioFromOSM.py` or SUMO's `netconvert` to convert the osm to `.net.xml`.
    1. See `./Map_Contruction/net_convert.sh`.
5. Open the `.net.xml` file in Netconvert, highlight `highway.service` edges, and add `pedestrian` and `taxi` to allowed vehicle types.
6. Copy raw gps-traces to `$Scenario_Dir/_Inputs/Traces/Original`.
7. Create a script in `$Src_Dir/data_processing_ev/scenario_initialisation/Data_Pre-Processing` to transform the gps-traces to be in the format of the original Stellenbosch data.
    1. Specifically, the data must be in CSV format with the below format.
    2. The headings in bold are mandatory.

|          | GPSID | **Time**                  | **Latitude** | **Longitude** | Altitude  | Heading   | Satellites | HDOP[^*] | AgeOfReading | DistanceSinceReading | **Velocity** |
|----------|-------|---------------------------|--------------|---------------|-----------|-----------|------------|----------|--------------|----------------------|--------------|
| Datatype | int   | str                       | float        | float         | int/float | int/float | int        | float    | int          | int                  | int          |
| units    | -     | 'yyyy/mm/dd HH:MM:ss APM' | [-]11.11111  | [-]11.11111   | meters    | degrees   | -          | meters   | minutes?     | meters               | km/h         |

8. Make sure the processed traces are in `$Scenario_Dir/_Inputs/Traces/Processed`.

[^*]: Horizontal Dilution of Precision. Lower is better.
