#!/bin/bash


SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
echo "Script directory: $SCRIPT_DIR"
data_dir=/work/pi_mserafini_umass_edu/dataset/gsplit/
run="sbatch --partition=cpu --cpus-per-task=8 --time=10:00:00 --mem=180GB"

# graph_name=$1
# data_dir=$2
# node_mode=$3
# edge_mode=$4
# bal=$5
for graph_name in friendster papers100M #products orkut
do
    for node_mode in uniform degree src dst input
    do
        for edge_mode in freq #uniform
        do
            for bal in bal xbal
            do
                job_name=${graph_name}_n${node_mode}_e${edge_mode}_${bal}
                outfile=$SCRIPT_DIR/logs/${job_name}.log
                $run --job-name=$job_name -o $outfile $SCRIPT_DIR/run_partition.sh $graph_name $data_dir $node_mode $edge_mode $bal
            done
        done
    done
done