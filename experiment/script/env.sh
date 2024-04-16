#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export data_dir=$(realpath $SCRIPT_DIR/../../dataset/graph/)
export python_dir=$(realpath $SCRIPT_DIR/..)

echo "Data directory: $data_dir"
echo "Data directory: $python_dir"
