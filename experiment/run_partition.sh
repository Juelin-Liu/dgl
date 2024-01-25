#!/bin/bash
work_dir=/scratch/workspace/juelinliu_umass_edu-metis/
script_path=/work/pi_mserafini_umass_edu/juelin/dgl/experiment/partition.py
graph_name=$1
data_dir=$2
node_mode=$3
edge_mode=$4
bal=$5

flags="--graph_name=${graph_name} --node_mode=${node_mode} --edge_mode=${edge_mode} --bal=${bal} --data_dir=${data_dir}"
cd $work_dir && python3 $script_path $flags