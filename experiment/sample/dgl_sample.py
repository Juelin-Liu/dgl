
import torch
from torch.multiprocessing import spawn
import torch.distributed as dist
from dgl.dev.dataloader import *
from dgl.dev import CudaProfilerStart, CudaProfilerStop
from utils import *

def dgl_sample(config: Config):
    graph, train_idx, valid_idx, test_idx = load_topo(config, is_pinned=True)
    partition_map = None
    if "random " not in config.partition_type:
        partition_map = load_partition_map(config)
    else:
        print(f"using random with {config.num_partition} partitions")
        v_num = graph.num_nodes()
        partition_map = torch.randint(low=0, high=config.num_partition, size=(v_num,), dtype=torch.int8)

    try:
        spawn(_dgl_sample, args=(config, graph, train_idx, partition_map),
              nprocs=config.world_size)
    except Exception as e:
        print(f"error encountered with {config=}:", e)
        exit(-1)


def _dgl_sample(rank: int, config: Config, graph: dgl.DGLGraph, train_idx: torch.Tensor, partition_map: torch.Tensor):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    sample_config = SampleConfig(rank=rank, num_partition=config.num_partition, batch_size=config.batch_size,
                                 world_size=config.world_size, mode=config.sample_mode, fanouts=config.fanouts, 
                                 reindex=True, drop_last=True,  node_rank=config.node_rank, num_nodes=config.num_nodes)

    dataloader = GraphDataloader(g=graph, target_idx=train_idx, config=sample_config)
    step = 0
    step_per_epoch = dataloader.max_step_per_epoch
    
    CudaProfilerStart()
    print(f"sampling on device: {device}", flush=True)
    num_edges = 0
    num_input_nodes = 0
    timer = Timer()
    for epoch in range(config.num_epoch):
        for input_nodes, output_nodes, blocks in dataloader:
            step += 1
            num_input_nodes += input_nodes.shape[0]
            for block in blocks:
                num_edges += block.num_edges()

            log_step(rank, epoch, step, step_per_epoch, timer)
    print(f"{rank=} {num_edges=} {num_input_nodes=} duration={timer.duration(2)} secs")
    dist.barrier()
    CudaProfilerStop()
    ddp_exit()
