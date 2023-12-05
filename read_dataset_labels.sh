#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <dataset_rel_path> <label_tensor_out_rel_path> <label_index>" >&2
  exit 1
fi

dataset_rel_path=$1
label_tensor_out_rel_path=$2
label_index=$3

./build/test/read_dataset_labels "$dataset_rel_path" "$label_tensor_out_rel_path" "$label_index"