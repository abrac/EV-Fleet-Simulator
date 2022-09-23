#!/bin/bash
osmium extract --strategy complete_ways --bbox <min_lon>,<min_lat>,<max_lon>,<max_lat> *.osm.pbf -o temp.osm.pbf
osmium cat temp.osm.pbf -o square_boundary.osm
rm temp.osm.pbf
