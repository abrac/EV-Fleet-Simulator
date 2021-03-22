#!/bin/bash
# Note: if the script doesn't work, try copy the code and run it from the terminal directly...

for f in ./Stop_Duration_PDFs/**/*.svg; do
    mkdir -p "./png-exports/Stop_Duration_Graphs/${${f#./Stop_Duration_PDFs}%/*}"
    inkscape "$f" -d 150 -o "./png-exports/Stop_Duration_Graphs/${${f#./Stop_Duration_PDFs/}%.*}.png"
done
