#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

dataset_dir=${SCRIPT_DIR}/dataset
mkdir -p $dataset_dir
pushd $dataset_dir
wget https://spara-artifact.s3.us-east-2.amazonaws.com/partition_map.tar.gz
tar -xvf partition_map.tar.gz && mv numpy partition_map

wget https://spara-artifact.s3.us-east-2.amazonaws.com/dataset.tar.gz
tar -xvf dataset.tar.gz && mv numpy graph
rm *.tar.gz
popd