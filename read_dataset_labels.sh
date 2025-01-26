#!/bin/bash

# Author: Junzhe Wang, Lam Tran
# Date: 15.12.2024, 26.01.2025
# Description: Script to process MNIST label dataset using the compiled C++ program.

# This script should read a dataset label into a tensor and pretty-print it into a text file...

# Exit if any commad fails.
set -e

# Argument validation.
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]
then
    echo "Usage: $0 <label_dataset_input> <label_tensor_output> [<label_index>]"
    exit 1
fi

label_dataset_input="$1"
label_tensor_output="$2"
label_index="${3:-}"

if [ ! -f "$label_dataset_input" ]
then
    echo "Error: Input file [$label_dataset_input] does NOT exist!"
    exit 1
fi

echo "MNIST dataset label processing..."

executable="./bin/read_dataset_labels"
if [ ! -x "$executable" ]
then
    echo "Error: Executable [$executable] not found or without executable permission!"
    exit 1
fi

if [ -n "$label_index" ] && [ "$label_index" -gt 0 ]
then
    "$executable" "$label_dataset_input" "$label_index" > "$label_tensor_output"
else
    "$executable" "$label_dataset_input" > "$label_tensor_output"
fi

if [ ! -f "$label_tensor_output" ]
then
    echo "Error: Failed to output label tensor file [$label_tensor_output]."
    exit 1
fi

echo "MNIST label dataset processed successfully."
echo "The label tensor output is written in [$label_tensor_output]."
exit 0

