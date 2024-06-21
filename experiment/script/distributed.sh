
#for graph in orkut friendster;
#do
#for model in sage gat;
#do	
#python3 train_main.py --system=split --model=${model} --fanout=15,15,15 --graph=${graph} --world_size=2 \
#	--data_dir=/data/gsplit --cache_size=8G --batch_size=512 --num_epoch=2 --devices=0,1 --nodes=2
#
#
#done
#done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

graph=papers100M
export PYTHONPATH=/home/sandeep/juelin/dgl/third_party/dist_cache/torch-quiver/srcs/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


for system in quiver;

do
for model in  sage gat;
do
	python3  ${python_dir}/train_main.py  --system=${system} --model=${model} --fanout=15,15,15 --graph=${graph} --world_size=4 \
	--data_dir=${data_dir}  --cache_size=6G --batch_size=1024 --num_epoch=3  --num-nodes=2 --node-rank=0 --log_file=distributed.csv


done
done
