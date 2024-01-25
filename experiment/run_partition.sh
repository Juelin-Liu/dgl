#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
echo "Script directory: $SCRIPT_DIR"
work_dir=/scratch/workspace/juelinliu_umass_edu-metis/

graph_name=$1
data_dir=$2
node_mode=$3
edge_mode=$4
bal=$5

outfile=$SCRIPT_DIR/logs/${graph_name}_n${node_mode}_e${edge_mode}_${bal}.txt

flags="--graph_name=${graph_name} --node_mode=${node_mode} --edge_mode=${edge_mode} --bal=${bal} --data_dir=${data_dir}"
echo partition.py with $flags
cd $work_dir && python3 $SCRIPT_DIR/partition.py $flags >> $outfile 2>&1