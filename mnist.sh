#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <input_config>" >&2
  exit 1
fi

input_config=$1

export NUM_CORES=$(nproc --all)
OMP_NUM_THREADS=$NUM_CORES ./build/app/mnist $input_config