#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
echo "Script directory: $SCRIPT_DIR"
# data_dir=/work/pi_mserafini_umass_edu/dataset/gsplit/
data_dir=/data/juelin/dataset/gsplit
# run="sbatch --partition=gpu --gpus=2080ti:4 --cpus-per-gpu=2 --time=00:10:00 --mem=96GB"
for graph_name in papers100M friendster products orkut 
do
    for node_mode in uniform degree src dst input
    do
        for edge_mode in freq uniform
        do
            for bal in bal xbal
            do
                job_name=${graph_name}_n${node_mode}_e${edge_mode}_${bal}_cnts
                outfile=$SCRIPT_DIR/logs/${job_name}.log
                # $run --job-name=$job_name -o $outfile $SCRIPT_DIR/run_simu.sh $graph_name $data_dir $node_mode $edge_mode $bal
                python3 simu.py --graph_name=$graph_name --data_dir=$data_dir --node_mode=$node_mode --edge_mode=$edge_mode --bal=$bal >> $outfile
            done
        done
    done
done