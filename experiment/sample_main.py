import argparse
import os
from utils import get_partition_type, Config

def get_parser():
    parser = argparse.ArgumentParser(description='local run script')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--num_epoch', default=5, type=int, help='Number of epochs to be sampled (default 5)')
    parser.add_argument('--fanouts', default="15,15,15", type=str, help='Input fanouts (15,15,15)')
    parser.add_argument('--graph_name', default="products", type=str, help="Input graph name", choices=["products", "papers100M", "orkut", "friendster"])
    parser.add_argument('--system', default="split", type=str, help="System", choices=["dgl", "split"])
    parser.add_argument('--sample_mode', default="uva", type=str, help="Sample mode", choices=["uva", "gpu"])
    parser.add_argument('--data_dir', default="/data/juelin/dataset/gsplit", type=str, help="Input graph directory")
    parser.add_argument('--world_size', default=4, type=int, help='Number of GPUs')
    parser.add_argument('--nmode', default="dst", type=str, help="Node weight configuraion", choices=["uniform", "degree", "src", "dst", "input", "random"] )
    parser.add_argument('--emode', default="freq", type=str, help="Edge weight configuraion", choices=["uniform", "freq", "random"])
    parser.add_argument('--bal', default="xbal", type=str, help='Balance target idx on each partition or not', choices=["bal", "xbal"])
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    print(f"{args=}")
    graph_name = str(args.graph_name)
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    nmode = args.nmode
    emode = args.emode
    bal = args.bal
    system = args.system
    sample_mode = args.sample_mode
    partition_type = get_partition_type(nmode, emode, bal)
    
    fanouts = args.fanouts.split(',')
    for idx, fanout in enumerate(fanouts):
        fanouts[idx] = int(fanout)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, "logs/exp.csv")
    
    config = Config(graph_name=graph_name,
                    world_size=4,
                    num_partition=4,
                    num_epoch=num_epoch,
                    fanouts=fanouts,
                    batch_size=batch_size,
                    system=system,
                    model="none",
                    cache_size="0GB",
                    hid_size=256,
                    log_path=log_path,
                    data_dir=data_dir,
                    nvlink=False,
                    partition_type=partition_type,
                    sample_mode=sample_mode)

    if config.system == "split":
        from sample.split_sample import split_sample
        split_sample(config)
    elif config.system == "dgl":
        from sample.dgl_sample import dgl_sample
        dgl_sample(config)