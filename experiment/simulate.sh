#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
data_dir=$SCRIPT_DIR/../dataset/graph/
echo "Data directory: $data_dir"

for graph_name in products papers100M orkut friendster
do
    for node_weight in uniform degree dst src input
    do
        for edge_weight in uniform freq
        do
            for bal in bal xbal
            do
                python3 simulate_main.py --graph_name=$graph_name --data_dir=$data_dir --node_weight=$node_weight --edge_weight=$edge_weight --bal=$bal
            done
        done
    done
done