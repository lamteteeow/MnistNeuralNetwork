#!/bin/bash

# Author: Junzhe Wang
# Date: 15.12.2024
# Description: Script to process MNIST dataset images using the compiled C++ program.

# This script should read a dataset image into a tensor and pretty-print it into a text file...

# Exit if any commad fails.
set -e

# Argument validation.
if [ "$#" -ne 2 ]
then
    echo "Usage: ./$0 <input> <output>"
    exit 1
fi

input="$1"
output="$2"

if [ ! -f "$input" ]
then
    echo "Error: Input file [$input] does NOT exist!"
    exit 1
fi

echo "MNIST dataset image processing..."

executable="./bin/read_dataset_images"
if [ ! -x "$executable" ]
then
    echo "Error: Executable [$executable] not found or without executable permission!"
    exit 1
fi

"$executable" "$input" > "$output"
if [ ! -f "$output" ]
then
    echo "Error: Failed to create output file [$output]."
    exit 1
fi

echo "MNIST dataset images processed successfully."
echo "The output is written in [$output]."
exit 0

