#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CUR_DIR=$SCRIPT_DIR

source "${CUR_DIR}/../script/env.sh"

python_dir=$CUR_DIR

for graph_name in products papers100M orkut friendster; do
# for graph_name in friendster; do
    for node_weight in uniform degree; do #dst src input
       for edge_weight in uniform; do
           for bal in bal xbal; do
               python3 ${python_dir}/get_partition.py --graph_name=$graph_name --data_dir=$data_dir --node_weight=$node_weight --edge_weight=$edge_weight --bal=$bal
           done
       done
    done

    for node_weight in dst; do #src input
        for edge_weight in freq uniform; do
            for bal in xbal; do #bal
                python3 ${python_dir}/get_partition.py --graph_name=$graph_name --data_dir=$data_dir --node_weight=$node_weight --edge_weight=$edge_weight --bal=$bal
            done
        done
    done
done
