#!/bin/bash

python_script=$1
graph_name=$2
data_dir=$3
node_mode=$4
edge_mode=$5
bal=$6
flags="--graph_name=${graph_name} --node_mode=${node_mode} --edge_mode=${edge_mode} --bal=${bal} --data_dir=${data_dir}"
echo "Flags: $flags"

# work_dir=${TMPDIR}
work_dir=/scratch/workspace/juelinliu_umass_edu-metis/
python_path=$(which python)

echo "Python Path: ${python_path}"
echo "Working Dir: ${work_dir}"

cd $work_dir && python3 $python_script $flags