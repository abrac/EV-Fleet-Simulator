#!/bin/bash
read -p "Does the scenario have have left-hand traffic? [y]/n  " LEFTHAND
read -p "Do you want to import elevation data from the OSM file? y/[n]  " ELEVATION

echo "Converting OSM file to SUMO Network..."

if [[ ${LEFTHAND,,} == "n" ]]
then
    echo "You selected right-hand traffic."

    if [[ ${ELEVATION,,} == "y" ]]
    then
        echo "You opted to import elevation data."
        netconvert --osm square_boundary.osm  \
            --type-files ./osmNetconvert_Africa.typ.xml \
            --geometry.remove \
            --ramps.guess \
            --junctions.join \
            --tls.guess-signals --tls.discard-simple --tls.join \
            --tls.default-type actuated \
            --log netconvert_errors.log.txt \
            --osm.elevation \
            -o ../square_boundary.net.xml
    else
        echo "You opted to ignore elevation data."
        netconvert --osm square_boundary.osm  \
            --type-files ./osmNetconvert_Africa.typ.xml \
            --geometry.remove \
            --ramps.guess \
            --junctions.join \
            --tls.guess-signals --tls.discard-simple --tls.join \
            --tls.default-type actuated \
            --log netconvert_errors.log.txt \
            -o ../square_boundary.net.xml
    fi
else
    echo "You selected left-hand traffic."

    if [[ ${ELEVATION,,} == "y" ]]
    then
        echo "You opted to import elevation data."
        netconvert --osm square_boundary.osm  \
            --type-files ./osmNetconvert_Africa.typ.xml \
            --geometry.remove \
            --ramps.guess \
            --junctions.join \
            --tls.guess-signals --tls.discard-simple --tls.join \
            --tls.default-type actuated \
            --lefthand \
            --log netconvert_errors.log.txt \
            --osm.elevation \
            -o ../square_boundary.net.xml
    else
        echo "You opted to ignore elevation data."
        netconvert --osm square_boundary.osm  \
            --type-files ./osmNetconvert_Africa.typ.xml \
            --geometry.remove \
            --ramps.guess \
            --junctions.join \
            --tls.guess-signals --tls.discard-simple --tls.join \
            --tls.default-type actuated \
            --lefthand \
            --log netconvert_errors.log.txt \
            -o ../square_boundary.net.xml
    fi
fi


# Rationalle behind command options:
#   --osm: The OpenStreetMap file.
#   --type-files: A type file defines for each road-type, what vehicles are
#       allowed, what the speed limit is, etc.
#   --geometry.remove: Simplifies the network (saving space) without changing
#       topology.
#   --ramps.guess: Acceleration/Deceleration lanes are often not included in
#       OSM data. This option identifies likely roads that have these
#       additional lanes and causes them to be added.
#   --junctions.join: See
#       https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html#junctions
#   --tls.guess-signals, --tls.discard-simple, --tls.join: See
#       https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html#traffic_lights
#   --tls.default-type actuated: Default static traffic lights are defined
#       without knowledge about traffic patterns and may work badly in high
#       traffic.
#   --left-handed: Informs netconvert that in the location, cars drive on the
#       left side of the road. TODO -- Provide the user with a prompt to
#       enable/disable this flag.
#   --log netconvert_error_log.txt: Save the errors and warnings in a text
#       file.
