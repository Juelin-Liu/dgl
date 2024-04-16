# How to Prepare Dataset
The following instructions will download the datasets from their original URLs and convert them into NumPy format. 
Guidance on how to generate node/edge weights for Metis partitioning is also provided.

## Download Raw Dataset from SNAP
```bash
sudo apt install aria2 -y
./download.sh
```
The download script will download and extract the raw data in `../../dataset/raw` directory.

## Convert Datasets to Numpy Format
```bash
python3 get_npgraph.py --data_dir=../../dataset/raw --graph_name=products
```
This script will convert the graph to the NumPy format and generate the train, valid, test index split.

## Generate Frequency Weight
For example, to generate node and edge weights for products, run:
```bash
python3 get_weight.py --data_dir=../../dataset/raw --graph_name=products
```
## Generate Partition Maps
For detailed usage:
```bash
python3 get_partition.py --help
```

It has six input parameters:
`--graph_name`: input graph name

`--data_dir`: input data directory

`--num_partition`: number of partitions in the result

`--node_weight`: weights assigned to each node. Valid options include: `[uniform,degree,src,dst,input]`.
1. `uniform`: each node has the same weight 1.
2. `degree`: each node has a weight equal to its degree.
3. `src`: each node has a weight equal to the number of times it is sampled as a source node during `get_weight.py`.
4. `dst`: each node has a weight equal to the number of times it is sampled as a destination node during `get_weight.py`.
5. `input`: each node has a weight equal to the number of times it is sampled as an input node during `get_weight.py`.

`--edge_weight`: weights assigned to each edge. Valid options include: `[uniform,freq]`.
1. `uniform`: each edge has the same weight 1.
2. `freq`: each node has a weight equal to the number of times it is sampled during `get_weight.py`.

`--bal`: balance train, valid, test splits or not. Valid options include: `[bal,xbal]`.
1. `bal`: balance train, valid, test splits.
2. `xbal`: do not balance train, valid, test splits.

Example usage:
```bash
python3 get_partition.py --num_partition=4 --graph_name=products --data_dir=../../dataset/raw --node_mode=dst --edge_mode=freq --bal=xbal
```
The result will be saved to `../../dataset/raw/products/products_w4_ndst_efreq_xbal.npy`.

Notice that this step might take several days for large graphs like Papers100M and Friendster. We provide preprocessed partition maps for all the tested graphs. 

Alternatively, you can refer to this [repo](https://github.com/Juelin-Liu/npmetis) for a multi-thread / distributed version of Metis.