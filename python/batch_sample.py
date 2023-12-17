from dgl.dev import *
from dgl.utils import pin_memory_inplace
from sample_util import *
from torch.utils.data import DataLoader
import argparse, os

def bench(configs: list[Config], test_acc=False):
    for config in configs:
        assert(config.system == configs[0].system and config.graph_name == configs[0].graph_name)
    in_dir = os.path.join(configs[0].data_dir, configs[0].graph_name)
    train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=True)
    indptr, indices, _label = load_graph(in_dir, is32=True, wsloop=True)
    # feat, label, num_label = load_feat_label(in_dir)
    feat = torch.zeros(1)
    label = torch.zeros(1)
    # _feat = pin_memory_inplace(feat)
    # _label = pin_memory_inplace(label)
    rank = device = 0
    if "uva" in config.system:
        _indptr = pin_memory_inplace(indptr)
        _indices = pin_memory_inplace(indices)
    elif "gpu" in config.system:
        indptr = indptr.to(rank)
        indices = indices.to(rank)
        
    for config in configs:
        # Default settings
        try:
            train(rank, config, indptr, indices, train_idx, test_idx, valid_idx, feat, label)
        except Exception as e:
            if "out of memory" in str(e):
                print("out of memory for", config)
                write_to_csv(config.log_path, [config], [oom_profiler()])
                continue
            else:
                write_to_csv(config.log_path, [config], [empty_profiler()])
                with open(f"exceptions/{config.get_file_name()}", 'w') as fp:
                    fp.write(str(e))
                continue
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def train(rank: int, config: Config, indptr, indices, train_idx: Tensor, test_idx, valid_idx, feat, label):
    SetGraph(indptr, indices)
    SetFanout(config.fanouts)
    train_idx_loader = DataLoader(train_idx.to(rank), batch_size=config.batch_size * config.pool_size, shuffle=True)
    print("start batched sampling")
    timer = Timer()
    num_edges = 0
    start_id = 0
    for epoch in range(config.num_epoch):
        for seeds in train_idx_loader:
            end_id = SampleBatches(seeds, config.batch_size, config.replace, config.batch_layer)
            for batch_id in range(start_id, end_id):
                input_nodes, output_nodes, blocks = GetBlocks(batch_id)
                for block in blocks:
                    num_edges += block.num_edges()
                    
            start_id = end_id


    print(f"batch {num_edges=}")
    duration = timer.duration()
    sampling_time = duration
    profiler = Profiler(num_epoch=config.num_epoch, duration=duration, sampling_time=sampling_time, feature_time=0.0, forward_time=0.0, backward_time=0.0, test_acc=0)
    write_to_csv(config.log_path, [config], [profiler])
    
def get_configs(data_dir, graph_name, batch_size, system, log_path, pool_size, batch_layer, replace):
    fanouts = [[10, 10, 10], [15, 15, 15], [20, 20, 20]]
    # fanouts = [[20, 20, 20]]
    configs = []
    for fanout in fanouts:
        config = Config(graph_name=graph_name, 
                        world_size=1, 
                        num_epoch=1, 
                        fanouts=fanout, 
                        batch_size=batch_size, 
                        system=system, 
                        model="sage",
                        hid_size=128, 
                        cache_size=0, 
                        log_path=log_path,
                        data_dir=data_dir,
                        pool_size=pool_size,
                        batch_layer=batch_layer,
                        replace=replace)
        configs.append(config)
    return configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--graph_name', default="ogbn-products", type=str, help="Input graph name any of ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']", choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M'])
    parser.add_argument('--data_dir', default="/mnt/homes/juelinliu/dataset/OGBN/processed", type=str, help="data directory")
    parser.add_argument('--epoch', default=1, type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=256, type=int, help='Input batch size on each device (default: 256)')
    parser.add_argument('--pool_size', default=1, type=int, help='pool size on each device (default: 1)')
    parser.add_argument('--batch_layer', default=0, type=int, help='at which layer starts to use batch loading')
    parser.add_argument('--replace', default=True, type=bool, help='use replace sampling or not')
    parser.add_argument('--system', default="", type=str, help='name of system', choices=["batch", "dgl-uva", "dgl-gpu", "base-uva", "base-gpu"])
    args = parser.parse_args()
    args = parser.parse_args()
    pool_size=int(args.pool_size)
    batch_size=int(args.batch_size)
    replace=bool(args.replace)
    graph_name=str(args.graph_name)
    data_dir=str(args.data_dir)
    batch_layer=int(args.batch_layer)
    system=str(args.system)
    print(f"{system=}")
    cur_dir = os.getcwd()
    log_path = os.path.join(cur_dir, "exp.csv")
    configs = get_configs(data_dir, graph_name, batch_size, system, log_path, pool_size, batch_layer, replace)
    bench(configs)