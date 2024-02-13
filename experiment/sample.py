from dgl.dev import *
from node.utils import load_topo, Config, Timer
import argparse
import torch

def get_parser():
    parser = argparse.ArgumentParser(description='local run script')
    parser.add_argument('--batch_size', default=256, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--num_epoch', default=1, type=int, help='Number of epochs to be sampled (default 20)')
    parser.add_argument('--use_bitmap', default=0, type=int, help='use bitmap or not')
    parser.add_argument('--fanouts', default="15,15,15", type=str, help='Input fanouts (15,15,15)')
    parser.add_argument('--graph_name', default="products", type=str, help="Input graph name", choices=["products", "papers100M", "orkut", "friendster"])
    parser.add_argument('--data_dir', default="/data/juelin/dataset/gsplit", type=str, help="Input graph directory")
    # parser.add_argument('--world_size', default=4, type=int, help='Number of GPUs')
    parser.add_argument('--mode', default="uva", type=str, help="Graph data configuraion", choices=["uva", "gpu"] )
    # parser.add_argument('--edge_mode', default="uniform", type=str, help="Edge weight configuraion", choices=["uniform", "freq", "random"])
    # parser.add_argument('--bal', default="bal", type=str, help='Balance target idx on each partition or not', choices=["bal", "xbal"])
    return parser.parse_args()


def get_size(s):
    return f"{round(s / 1024 / 1024, 1)}MB"
    
if __name__ == "__main__":
    args = get_parser()
    graph_name = args.graph_name
    batch_size = args.batch_size
    data_dir = args.data_dir
    use_bitmap = args.use_bitmap
    fanouts = args.fanouts.split(',')
    for idx, fanout in enumerate(fanouts):
        fanouts[idx] = int(fanout)
    mode = args.mode

    config = Config(graph_name=graph_name,
                    world_size=1,
                    num_epoch=1,
                    fanouts=fanouts,
                    batch_size=batch_size,
                    system="dgl",
                    model="sage",
                    cache_size="0GB",
                    hid_size=256,
                    log_path="./log",
                    data_dir=data_dir)

    graph, train_idx, valid_idx, test_idx = load_topo(config, False)
    if mode == "uva":
        graph = graph.pin_memory_()
    elif mode == "gpu":
        graph = graph.to(0)

    sample_config = SampleConfig(rank=0, batch_size=config.batch_size * config.world_size, world_size=config.world_size, mode=mode, fanouts=config.fanouts, reindex=True, drop_last=True)
    dataloader = GraphDataloader(graph, train_idx, sample_config)
    UseBitmap(use_bitmap)

    timer = Timer()
    num_edges = 0
    num_input_nodes = 0
    for input_nodes, output_nodes, blocks in dataloader:
        num_input_nodes += input_nodes.shape[0]
        for block in blocks:
            num_edges += block.num_edges()
        
    duration = timer.duration(2)
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()

    print(f"{graph_name=} {use_bitmap=} {duration=}secs reserved = {get_size(reserved)} allocated = {get_size(allocated)} {num_edges=} {num_input_nodes=}")
