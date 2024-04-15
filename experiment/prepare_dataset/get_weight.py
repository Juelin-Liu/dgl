import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import torch.distributed as dist
import torch
from torch.multiprocessing import spawn
from utils import *
from dgl.dev import *
from dgl.dev.dataloader import SampleConfig, GraphDataloader
from dgl.dev.coo2csr import Increment
    
def freq(config: Config):
    graph, train_idx, valid_idx, test_idx = load_topo(config, is_pinned=True)
    try:
        spawn(_freq, args=(config, graph, train_idx), nprocs=config.world_size)
    except Exception as e:
        print(f"error encountered with {config=}:", e)
        exit(-1)
            
def _freq(rank: int, config: Config, graph: dgl.DGLGraph, train_idx: torch.Tensor):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    e2eTimer = Timer()
    if rank == 0:
        print(config)
    mode = "uva"
    graph = graph.pin_memory_()    
    sample_config = SampleConfig(rank=rank, batch_size=config.batch_size * config.world_size, world_size=config.world_size, mode=mode, fanouts=config.fanouts, reindex=False)
    dataloader = GraphDataloader(graph, train_idx, sample_config)
    step = 0
    step_per_epoch = dataloader.max_step_per_epoch
    
    print(f"sampling on device: {device}")        
    timer = Timer()
    epoch_num = config.num_epoch
    min_epoch_num = 1500 // step_per_epoch + 1
    # max_epoch_num = 50000 // step_per_epoch + 1
    epoch_num = max(min_epoch_num, epoch_num)
    # epoch_num = min(epoch_num, max_epoch_num)
    v_num = graph.num_nodes()
    e_num = graph.num_edges()
    
    input_node_weight = torch.zeros((v_num,),dtype=torch.int32, device=rank)
    src_node_weight = torch.zeros((v_num,),dtype=torch.int32, device=rank)
    dst_node_weight = torch.zeros((v_num,),dtype=torch.int32, device=rank)
    edge_weight = torch.zeros((e_num,), dtype=torch.int32, device=rank) # TODO enable int16
    
    for epoch in range(epoch_num):
        for input_nodes, output_nodes, blocks in dataloader:
            step += 1
            
            Increment(input_node_weight, input_nodes)
            for block in blocks:
                src, dst = block.all_edges()
                edge_id = block.edata["_ID"]
                Increment(dst_node_weight, dst)
                Increment(src_node_weight, src)
                Increment(edge_weight, edge_id)
                
            log_step(rank, epoch, step, step_per_epoch, timer)

    dist.barrier()
    
    dist.all_reduce(input_node_weight)
    dist.all_reduce(src_node_weight)
    dist.all_reduce(dst_node_weight)
    dist.all_reduce(edge_weight)
    
    if rank == 0:
        out_dir = os.path.join(config.data_dir, config.graph_name)
        print("saving to", out_dir)
        save_numpy(input_node_weight.type(torch.int64), f"{out_dir}/input_node_weight.npy")
        save_numpy(src_node_weight.type(torch.int64), f"{out_dir}/src_node_weight.npy")
        save_numpy(dst_node_weight.type(torch.int64), f"{out_dir}/dst_node_weight.npy")
        save_numpy(edge_weight.type(torch.int64), f"{out_dir}/edge_weight.npy")
    ddp_exit()

if __name__ == "__main__":
    
    args = get_args()
    
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
    cfg = Config(graph_name=graph_name,
                       world_size=4,
                       num_epoch=num_epoch,
                       num_partition=1,
                       fanouts=fanouts,
                       batch_size=batch_size,
                       system="dgl-sample",
                       model="none",
                       cache_size="0GB",
                       hid_size=256,
                       log_path=log_path,
                       data_dir=data_dir)
           
    freq(cfg)