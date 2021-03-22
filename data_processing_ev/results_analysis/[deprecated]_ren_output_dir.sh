#!/bin/bash

error=0
if [ $# -gt 2 ]
then
    dir_name_old=$1
    dir_name_new=$2
    dir_parent=$3
elif [ $# -eq 2 ]
then
    dir_name_old=$1
    dir_name_new=$2
    dir_parent="Network_*"
elif [ $# -eq 1 ]
then
    dir_name_old="Output"
    dir_name_new=$1
    dir_parent="Network_*"
else
    error=1
    echo "Error, not enough input arguments. Aborting."
fi

if [ ${error} != 1 ]
then
    printf "Rename \"${dir_name_old}\" to \"${dir_name_new} in in the following directories? [y/N] \n$(ls -d ${dir_parent}) \n    "
    read input
    case $input in
        [yY] | [yY][eE][sS])
            for dir in $(ls -d ${dir_parent})
            do
                mv ${dir}/${dir_name_old} ${dir}/${dir_name_new}
            done
            ;;
        *)
            echo aborting...
            ;;
    esac
fi
