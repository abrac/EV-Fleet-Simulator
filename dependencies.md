> Note: Version numbers are just the ones that I tested the program with. Other versions may also work.
> But many packages and software introduce breaking changes between releases. So be aweare of that.

> Note: Items marked with a `*` are mandatory. The software may still work with the other items.


Software
--------

| Name       | Version | Description                 |
|------------|:-------:|-----------------------------|
| *SUMO [^1] |  1.8.0  | Traffic mobility simulator. |
| *Python    |   3.8   | A dangerous snake.          |
| *Bash      |    5    | The Linux terminal.         |


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

| Name            | Version | Description                                                                   |
|-----------------|:-------:|-------------------------------------------------------------------------------|
| *tqdm           |         | Progress bars.                                                                |
| *matplotlib     |         | Plots.                                                                        |
| *pandas         |         | Structuring & manipulating data.                                              |
| *hdbscan        |         |                                                                               |
| *numpy          |         |                                                                               |
| *scipy          |         |                                                                               |
| *scikit-learn   |         |                                                                               |
| *folium         |         | Map visualisation library.                                                    |
| *haversine      |  2.3.1  | Calculating distance between GPS coordinates.                                 |
| *rtree          |         | Used by SUMO for finding lanes that are closest to a specified geo-coordinate |
| memory_profiler |         |                                                                               |

[^1]: Make sure that libsumo is compiled with SUMO. The Ubuntu PPA does not include it by default.
