> Note: Version numbers are just the ones that I tested the program with. Other 
> versions may also work. But many packages and software introduce breaking 
> changes between releases. So be aweare of that.

> Note: Packages marked with a `*` are mandatory. The other packages are 
> recommended, but the software may work without them. Packages marked with a
> `+` are conditionally required. I.e., they are only required if performing 
> specific tasks. However, they are highly recommended.


Software
--------

    |-------------|------------|---------------------------------------------|
    | Name        |   Version  | Description                                 |
    |=============|:==========:|=============================================|
    | *SUMO [^1]  |    1.8.0   | Traffic mobility simulator.                 |
    |-------------|------------|---------------------------------------------|
    | *Python     |     3.8    | An awesome programming language named after |
    |             |            | a dangerous snake.                          |
    |-------------|------------|---------------------------------------------|
    | *Bash       |      5     | The terminal used in Linux and MacOS.       |
    |             |            | Windows can run Bash through WSL.           |
    |-------------|------------|---------------------------------------------|
    | *SAM        | 2020.11.29 | Simulator of Renewable-Energy Generators    |
    |             |            | (Solar panels, wind-turbines, etc.).        |
    |-------------|------------|---------------------------------------------|
    | *R          |    4.0.4   | R programming language.                     |
    |-------------|------------|---------------------------------------------|
    | *OSMConvert |   0.8.10   | Software for cropping OpenStreetMap files.  |
    |             |            | Not available on Mac. For Mac, use Osmosis  |
    |             |            | or Osmium-Tool.                             |
    |-------------|------------|---------------------------------------------|

    [^1]: Make sure that libsumo is compiled with SUMO. The Ubuntu PPA does not 
          include it by default. Libsumo is required by *Step 4 (Routing)*. The 
          other steps will work without libsumo. I think the Windows binaries 
          released by sumo are compiled together with libsumo.

SUMO Dependencies
-----------------
Ubuntu packages:
- *cmake 
- *python3 
- *g++ 
- *libxerces-c-dev 
- *libfox-1.6-dev 
- *libgdal-dev 
- *libproj-dev 
- *libgl2ps-dev 
- *python3-dev 
- *swig  <!-- I think... -->


Python Packages
---------------

    |-----------------|---------|-----------------------------------------|
    | Name            | Version | Description                             |
    |=================|:=======:|=========================================|
    | *tqdm           |         | Progress bars.                          |
    |-----------------|---------|-----------------------------------------|
    | *matplotlib     |         | Plots.                                  |
    |-----------------|---------|-----------------------------------------|
    | *pandas         |         | Structuring & manipulating data.        |
    |-----------------|---------|-----------------------------------------|
    | *hdbscan        |         |                                         |
    |-----------------|---------|-----------------------------------------|
    | *numpy          |         |                                         |
    |-----------------|---------|-----------------------------------------|
    | *scipy          |         |                                         |
    |-----------------|---------|-----------------------------------------|
    | *scikit-learn   |         |                                         |
    |-----------------|---------|-----------------------------------------|
    | *folium         |         | Map visualisation library.              |
    |-----------------|---------|-----------------------------------------|
    | *haversine      |  2.3.1  | Calculating distance between GPS        |
    |                 |         | coordinates.                            |
    |-----------------|---------|-----------------------------------------|
    | *rtree          |         | Used by SUMO for finding lanes that are |
    |                 |         | closest to a specified geo-coordinate   |
    |-----------------|---------|-----------------------------------------|
    | memory_profiler |         |                                         |
    |-----------------|---------|-----------------------------------------|


R packages
----------

    |----------------|---------|--------------------------------------------|
    | Name           | Version | Description                                |
    |================|:=======:|============================================|
    | +gtfs2gps [^1] |  1.6.0  | Converts GTFS public transport data to GPS |
    |                |         | traces.                                    |
    |----------------|---------|--------------------------------------------|

    [^1]: Required if using GTFS data inputs.


