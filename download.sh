#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/experiment/script/env.sh"

# mkdir -p $data_dir
# pushd $data_dir

# wget https://spara-artifact.s3.us-east-2.amazonaws.com/dataset.tar.gz
# tar -xvf dataset.tar.gz && mv numpy graph

# wget https://spara-artifact.s3.us-east-2.amazonaws.com/partition_map.tar.gz
# tar -xvf partition_map.tar.gz && mv numpy ./graph/partition_map

# rm *.tar.gz
# popd