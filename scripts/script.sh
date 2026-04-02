#!/bin/bash
#
# Watches the current directory for .ft3 files, moves them to ./input,
# runs the C++ converter, and writes results to ./output.
#
# Usage: Run from the directory containing .ft3 files.
#   bash script.sh

CONVERTER="$(dirname "$0")/../bin/converter"

mkdir -p ./input ./output

while true; do
    file_count=$(find . -maxdepth 1 -name '*.ft3' | wc -l)
    echo "Number of ft3 files: $file_count"

    if [[ $file_count -gt 1 ]]; then
        echo "Initiate conversion"
        move_file_count=$((file_count - 1))
        echo "Number of files to move: $move_file_count"

        for ft_file in $(ls -lt *.ft3 | awk '{print $9}' | tail -"${move_file_count}"); do
            echo "Moving file $ft_file"
            mv "$ft_file" ./input/
        done

        # Run the C++ converter
        "$CONVERTER" --input ./input --output ./output --lmax 14

        rm ./input/*
    else
        echo "Waiting for sufficient files to process."
    fi

    sleep 5
done
