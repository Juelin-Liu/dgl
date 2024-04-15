#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# DATA DIRECTORY ROOT
data_dir=$SCRIPT_DIR/../../dataset/raw # CHANGE THIS TO YOUR PREFERRED LOCATIONS

echo "Dataset directory: $data_dir"

mkdir -p $data_dir
pushd $data_dir 

aria2c -x 16 -s 16 http://snap.stanford.edu/ogb/data/nodeproppred/products.zip
aria2c -x 16 -s 16 https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz
aria2c -x 16 -s 16 http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip
aria2c -x 16 -s 16 https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz

# wget http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip
# wget http://snap.stanford.edu/ogb/data/nodeproppred/products.zip
# wget https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
# wget https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz

unzip products.zip && mv products ogbn_products
gzip -d com-orkut.ungraph.txt.gz && mkdir -p orkut && mv com-orkut.ungraph.txt orkut/orkut.txt
unzip papers100M-bin.zip && mv papers100M-bin ogbn_papers100M
gzip -d com-friendster.ungraph.txt.gz && mkdir -p friendster && mv com-friendster.ungraph.txt friendster/friendster.txt

rm *.zip
rm *.gz

popd
