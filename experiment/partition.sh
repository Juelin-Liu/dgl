#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
echo "Script directory: $SCRIPT_DIR"
data_dir=/work/pi_mserafini_umass_edu/dataset/gsplit/
run="sbatch --partition=cpu --cpus-per-task=2 --time=10:00:00"
world_size=8
for graph_name in friendster papers100M orkut products 
do
    if [[ "$graph_name" == friendster ]]; then
        mem="240GB"
    elif [[ "$graph_name" == papers100M ]]; then
        mem="160GB"
    else
        mem="36GB"
    fi
    for node_mode in uniform degree #dst src input
    do
        for edge_mode in uniform
        do
            for bal in bal
            do
                job_name=${graph_name}_w${world_size}_n${node_mode}_e${edge_mode}_${bal}
                outfile=$SCRIPT_DIR/logs/${job_name}.log
                bash_script=$SCRIPT_DIR/run_partition.sh
                python_script=$SCRIPT_DIR/partition.py
                $run --mem=$mem --job-name=$job_name --output $outfile $bash_script $python_script $graph_name $data_dir $node_mode $edge_mode $bal $world_size
            done
        done
    done

    for node_mode in dst #src input
    do
        for edge_mode in freq uniform
        do
            for bal in xbal
            do
                job_name=${graph_name}_w${world_size}_n${node_mode}_e${edge_mode}_${bal}
                outfile=$SCRIPT_DIR/logs/${job_name}.log
                bash_script=$SCRIPT_DIR/run_partition.sh
                python_script=$SCRIPT_DIR/partition.py
                $run --mem=$mem --job-name=$job_name --output $outfile $bash_script $python_script $graph_name $data_dir $node_mode $edge_mode $bal $world_size
            done
        done
    done
done
