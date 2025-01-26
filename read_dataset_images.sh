#!/bin/bash

# Author: Junzhe Wang, Lam Tran
# Date: 26.01.2025
# Description: Script to process MNIST image dataset using the compiled C++ program.

# This script should read a dataset image into a tensor and pretty-print it into a text file...

# Exit if any commad fails.
set -e

# Argument validation.
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]
then
    echo "Usage: $0 <image_dataset_input> <image_tensor_output> [<image_index>]"
    exit 1
fi

image_dataset_input="$1"
image_tensor_output="$2"
image_index="${3:-}"

if [ ! -f "$image_dataset_input" ]
then
    echo "Error: Input file [$image_dataset_input] does NOT exist!"
    exit 1
fi

echo "MNIST dataset image processing..."

executable="./bin/read_dataset_images"
if [ ! -x "$executable" ]
then
    echo "Error: Executable [$executable] not found or without executable permission!"
    exit 1
fi

if [ -n "$image_index" ] && [ "$image_index" -gt 0 ]
then
    "$executable" "$image_dataset_input" "$image_index" > "$image_tensor_output"
else
    "$executable" "$image_dataset_input" > "$image_tensor_output"
fi

if [ ! -f "$image_tensor_output" ]
then
    echo "Error: Failed to output image tensor file [$image_tensor_output]."
    exit 1
fi

echo "MNIST image dataset processed successfully."
echo "The image tensor output is written in [$image_tensor_output]."
exit 0

