
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

graph=products
model=sage
python3 train_main.py --system=split --model=${model} --fanout=15,15,15 --graph=${graph} --world_size=1 \
	--data_dir=/data/gsplit --cache_size=8G --batch_size=512 --num_epoch=2 --devices=0,1 --num_nodes=2 --node_rank=0



