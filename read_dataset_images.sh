#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <dataset_rel_path> <image_tensor_out_rel_path> <image_index>" >&2
  exit 1
fi

dataset_rel_path=$1
image_tensor_out_rel_path=$2
image_index=$3

./build/test/read_dataset_images "$dataset_rel_path" "$image_tensor_out_rel_path" "$image_index"