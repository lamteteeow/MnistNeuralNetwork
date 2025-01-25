#!/bin/bash

# Author: Hamiz Ali
# Date: 24.01.2025
# Description: Script to train and evaluate neural network using the compiled C++ program.

# outputs log_file provided in the input.config


# Exit if any command fails
set -e

# Argument validation
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_config_file>"
    exit 1
fi

input_config_file="$1"


echo "Training and evaluating neural network..."

executable="./bin/neural_network"
if [ ! -x "$executable" ]
then
    echo "Error: Executable [$executable] not found or without executable permission!"
    exit 1
fi


"$executable" "$input_config_file"

if [ $? -eq 0 ]; then
    echo "Training and evaluation completed successfully. Log written to '$rel_path_log_file'."
else
    echo "Error: Training and evaluation failed. Check log for details."
fi
