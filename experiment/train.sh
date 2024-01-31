#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
echo "Script directory: $SCRIPT_DIR"
nvlink=0
MACHINE_NAME=power9
batch_size=1024
fanouts="15,15,15"
num_epoch=10
data_dir=/work/pi_mserafini_umass_edu/dataset/gsplit/
run="sbatch --partition=power9-gpu --gpus-per-node=v100:4 --mem=188GB --cpus-per-gpu=4 --time=02:00:00"
bash_script=$SCRIPT_DIR/run_train.sh
python_script=$SCRIPT_DIR/train.py

# python_script=$1
# nvlink=$2
# cache_size=$3
# data_dir=$4
# graph_name=$5
# num_epoch=$6
# batch_size=$7
# fanouts=$8
# system=$9
# model=$10
for graph_name in papers100M orkut friendster
do
    for model in sage gat
    do
        for system in dgl p3
        do
            # echo running experiment on $graph_name using $system $model
            # python train.py --num_epoch=${num_epoch} --data_dir=${data_dir} --system=${system} --model=${model} --nvlink=${nvlink} --graph_name=${graph_name} --batch_size=${batch_size} --fanouts=${fanouts}
            # flags="--nvlink=${nvlink} --data_dir=${data_dir} --graph_name=${graph_name} --num_epoch=${num_epoch} --batch_size=${batch_size} --fanouts=${fanouts} --system=${system} --model=${model}"
            job_name=${graph_name}-${system}-${model}
            outfile=$SCRIPT_DIR/logs/${job_name}.log
            # $run --job-name=$job_name -o $outfile $bash_script $python_script $flags
            cache_size="0GB"
            $run --job-name=$job_name --output $outfile $bash_script $python_script $data_dir $graph_name $num_epoch $batch_size $fanouts $system $model $nvlink $cache_size 
        done
        for cache_size in 0GB #1GB 2GB #3GB 4GB 5GB 6GB 7GB 8GB 9GB 10GB
        do 
            system=quiver
            # echo running experiment on $graph_name using $system $model with cache size $cache_size
            # flags="--nvlink=${nvlink} --cache_size=${cache_size} --data_dir=${data_dir} --graph_name=${graph_name} --num_epoch=${num_epoch} --batch_size=${batch_size} --fanouts=${fanouts} --system=${system} --model=${model}"
            job_name=${graph_name}-${system}-${model}-${cache_size}
            outfile=$SCRIPT_DIR/logs/${job_name}.log
            # $run --job-name=$job_name --output $outfile $bash_script $python_script $flags
            # python train.py --cache_size=${cache_size} --num_epoch=${num_epoch} --data_dir=${data_dir} --system=${system} --model=${model} --nvlink=${nvlink} --graph_name=${graph_name} --batch_size=${batch_size} --fanouts=${fanouts}
            $run --job-name=$job_name --output $outfile $bash_script $python_script  $data_dir $graph_name $num_epoch $batch_size $fanouts $system $model $nvlink $cache_size 
        done
    done
done
