from dgl.dev import *
from preprocess.freq import *
import argparse
def get_parser():
    parser = argparse.ArgumentParser(description='local run script')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--num_epoch', default=20, type=int, help='Number of epochs to be sampled (default 20)')
    parser.add_argument('--fanouts', default="15,15,15", type=str, help='Input fanouts (15,15,15)')
    parser.add_argument('--graph_name', default="products", type=str, help="Input graph name", choices=["products", "papers100M", "orkut", "friendster"])
    parser.add_argument('--data_dir', default="/data/juelin/dataset/gsplit", type=str, help="Input graph directory")
    parser.add_argument('--world_size', default=4, type=int, help='Number of GPUs')
    return parser

if __name__ == "__main__":
    
    args = get_parser().parse_args()
    
    print(f"{args=}")
    graph_name = str(args.graph_name)
    data_dir = args.data_dir
    batch_size = args.batch_size
    fanouts = args.fanouts.split(',')
    num_epoch = args.num_epoch
    for idx, fanout in enumerate(fanouts):
        fanouts[idx] = int(fanout)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, "logs/exp.csv")
    config = Config(graph_name=graph_name,
                       world_size=4,
                       num_epoch=num_epoch,
                       fanouts=fanouts,
                       batch_size=batch_size,
                       system="dgl-sample",
                       model="none",
                       cache_size="0GB",
                       hid_size=256,
                       log_path=log_path,
                       data_dir=data_dir)
           
    freq(config)