#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
echo "Script directory: $SCRIPT_DIR"
data_dir=/work/pi_mserafini_umass_edu/dataset/gsplit/
run="sbatch --partition=cpu --cpus-per-task=8 --time=10:00:00 --mem=180GB"

for graph_name in friendster papers100M products orkut
do
    for node_mode in uniform degree dst #src input
    do
        for edge_mode in freq uniform
        do
            for bal in bal xbal
            do
                job_name=${graph_name}_n${node_mode}_e${edge_mode}_${bal}
                outfile=$SCRIPT_DIR/logs/${job_name}.log
                bash_script=$SCRIPT_DIR/run_partition.sh
                python_script=$SCRIPT_DIR/partition.py
                $run --job-name=$job_name --output $outfile $bash_script $python_script $graph_name $data_dir $node_mode $edge_mode $bal
            done
        done
    done
done