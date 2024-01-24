#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
echo "Script directory: $SCRIPT_DIR"
work_dir=/data/juelin/project/scratch/workspace
for graph_name in products
do
    for node_mode in uniform # degree src dst input
    do
        for edge_mode in uniform # freq
        do
            for bal in bal # xbal
            do
                outfile=${graph_name}_n${node_mode}_e${edge_mode}_${bal}.txt
                flags="--graph_name=${graph_name} --node_mode=${node_mode} --edge_mode=${edge_mode} --bal=${bal}"

                echo run with $flags
                pushd $work_dir && python3 $SCRIPT_DIR/partition.py $flags > $outfile 2>&1
            done
        done
    done
done