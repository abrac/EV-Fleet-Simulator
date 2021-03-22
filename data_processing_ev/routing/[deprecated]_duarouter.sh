#!/bin/bash

# Uses duarouter to create edge-based trips from GPS-coordinate-based trips
# TODO Convert this to Python

ROOTDIR=$(pwd)
# Loop through each day-clustered taxi trace
for route_file in $ROOTDIR/Clusters/*/*.trip.xml; do
    subpath=$(basename $(dirname $route_file))  # Name of taxi
    trip_name=$(basename $route_file)  # Name of taxi with date of trip
    trip_name="${trip_name%.trip.xml}"  # Removing the ".trip.xml" from the filename
    trip_path="$ROOTDIR/Taxi_Routes/$subpath/$trip_name"  # The final path name
    mkdir -p $trip_path  # Try make the path. TODO `Continue` to next iteration if fails
    cd $trip_path  # Go to the new path.
    # Run `duaIterate` once on the file. This is a hack to convert GPS coordinates to
    #   edges. No duarouting is done yet. TODO Stop using duaIterate to convert your
    #   traces. Python will provide more control.
    python $SUMO_TOOLS/assign/duaIterate.py \
        -n $ROOTDIR/Network/osm.net.xml \
        -t $route_file \
        -l 1 \
        duarouter--keep-vtype-distributions true \
        duarouter--routing-threads 4 \
        sumo--threads 4
        #duarouter--output-prefix "${route_file}/"
    cd $ROOTDIR
done
