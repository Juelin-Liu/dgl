#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
echo "Script directory: $SCRIPT_DIR"

# DATA DIRECTORY ROOT
export data_dir=$SCRIPT_DIR/../dataset # CHANGE THIS TO YOUR PREFERRED LOCATIONS

mkdir -p $data_dir

aria2c -x 16 -s 16 http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip -d $data_dir
aria2c -x 16 -s 16 http://snap.stanford.edu/ogb/data/nodeproppred/products.zip -d $data_dir
aria2c -x 16 -s 16 https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz -d $data_dir
aria2c -x 16 -s 16 https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz -d $data_dir

# wget http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip -P $data_dir
# wget http://snap.stanford.edu/ogb/data/nodeproppred/products.zip -P $data_dir
# wget https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz -P $data_dir
# wget https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz -P $data_dir

pushd $data_dir && unzip -d papers100M-bin.zip && mv papers100M-bin ogbn_papers100M
pushd $data_dir && unzip -d products.zip && mv products ogbn_products
pushd $data_dir && gzip -d com-friendster.ungraph.txt.gz && mkdir -p friendster && mv com-friendster.ungraph.txt friendster/friendster.txt
pushd $data_dir && gzip -d com-orkut.ungraph.txt.gz && mkdir -p orkut && mv com-orkut.ungraph.txt orkut/orkut.txt