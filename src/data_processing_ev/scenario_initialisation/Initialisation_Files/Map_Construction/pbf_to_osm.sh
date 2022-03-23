#!/bin/bash
osmium extract --strategy complete_ways -bbox <min_lon>,<min_lat>,<max_lon>,<max_lat> *.osm.pbf -o extract.osm.pbf
osmium cat extract.osm.pbf -o square_boundary.osm
