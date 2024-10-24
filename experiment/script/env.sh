#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export dataset_dir=$(realpath $SCRIPT_DIR/../../dataset/)
export data_dir=$(realpath $dataset_dir/graph/)
export python_dir=$(realpath $SCRIPT_DIR/..)

echo "Dataset directory: $dataset_dir"
echo "Data directory: $data_dir"
echo "Python directory: $python_dir"
