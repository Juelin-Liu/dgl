#!/bin/bash


SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
echo "Script directory: $SCRIPT_DIR"
data_dir=/work/pi_mserafini_umass_edu/dataset/gsplit/

run="srun --partition=gpu --gpus=2080ti:1 --cpus-per-gpu=1 --mem=32GB"

# graph_name=$1
# data_dir=$2
# node_mode=$3
# edge_mode=$4
# bal=$5
for graph_name in products
do
    for node_mode in uniform degree src dst input
    do
        for edge_mode in uniform freq
        do
            for bal in bal xbal
            do                
                $run $SCRIPT_DIR/run_partition.sh $graph_name $data_dir $node_mode $edge_mode $bal &
                sleep 1
            done
        done
    done
done