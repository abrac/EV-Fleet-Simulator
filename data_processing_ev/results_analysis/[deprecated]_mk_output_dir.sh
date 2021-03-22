#!/bin/bash

if [ $# -gt 1 ]
then
    dir_name=$1
    dir_parent=$2
elif [ $# -eq 1 ]
then
    dir_name=$1
    dir_parent="Network_*"
else
    dir_name="Output"
    dir_parent="Network_*"
fi

printf "Make \"${dir_name}\" directories in the following folders? [y/N] \n$(ls -d ${dir_parent}) \n    "
read input
case $input in
    [yY] | [yY][eE][sS])
        for dir in $(ls -d ${dir_parent})
        do
            mkdir $dir/$dir_name
        done
        ;;
    *)
        echo aborting...
        ;;
esac
