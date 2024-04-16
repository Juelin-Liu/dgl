import torchmetrics.functional as MF
import torch.distributed as dist
import gc
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from utils import *
from dgl.dev.dataloader import *

def simulate(config: Config):
    graph, train_idx, _, _ = load_topo(config, is_pinned=True)
    partition_map = None
    if not "random" in config.partition_type:
        partition_map = load_partition_map(config)
        assert(partition_map.max() == config.world_size - 1)
        print(f"load partition map {config.partition_type}:", partition_map, torch.bincount(partition_map))
    else:
        print(f"using random with {config.num_partition} partitions")
        v_num = graph.num_nodes()
        partition_map = torch.randint(low=0, high=config.num_partition, size=(v_num,), dtype=torch.uint8)
        print(f"load partition map {config.partition_type}:", partition_map, torch.bincount(partition_map))
    try:
        spawn(_simulate, args=(config, graph, train_idx, partition_map), nprocs=config.world_size)
    except Exception as e:
        print(f"error encountered with {config=}:", e)
        exit(-1)
            
def _simulate(rank: int, config: Config, graph: dgl.DGLGraph, train_idx: torch.Tensor, partition_map: torch.Tensor):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    partition_map = partition_map.to(device)
    e2eTimer = Timer()
    if rank == 0:
        print(config)
    sample_config = SampleConfig(rank=rank, batch_size=config.batch_size, world_size=config.world_size, mode=config.sample_mode, reindex=False, fanouts=config.fanouts)
    dataloader = GraphDataloader(graph, train_idx, sample_config)
    step_per_epoch = dataloader.max_step_per_epoch
    step = 0
    
    print(f"sampling on device: {device}", flush=True)        
    timer = Timer()
    
    src_node_cnts = []
    dst_node_cnts = []
    input_node_cnts = []
    crs_edge_cnts = []
    loc_edge_cnts = []
    
    def get_edge_cnts(src, dst, mapping):
        src_partition = mapping[src]
        dst_partition = mapping[dst]
        loc_mask = src_partition == dst_partition
        crs_mask = src_partition != dst_partition
        loc_cnts = torch.bincount(mapping[src[loc_mask]], minlength=config.num_partition) # use the src/s partition id to map the local edge (identical)
        crs_cnts = torch.bincount(mapping[src[crs_mask]], minlength=config.num_partition) # use the dst's partition id to map the cross edge (subject to change)
        
        # if rank == 0:
        #     print(f"{src=} {dst=}")
        #     print(f"{src_partition=} {dst_partition=}")

        return loc_cnts, crs_cnts
    
    def get_node_cnts(nodes, mapping):
        return torch.bincount(mapping[nodes], minlength=config.num_partition)
    
    def lst_to_tensor(lst):
        return torch.stack(lst)
    
    for epoch in range(config.num_epoch):
        for input_nodes, output_nodes, blocks in dataloader:
            step += 1
            
            input_node_cnts.append(get_node_cnts(input_nodes, partition_map))
            for block in blocks:
                src, dst = block.all_edges()
                loc_cnts, crs_cnts = get_edge_cnts(src, dst, partition_map)
                loc_edge_cnts.append(loc_cnts)
                crs_edge_cnts.append(crs_cnts)
                
                unique_src = block.srcnodes()
                src_node_cnts.append(get_node_cnts(unique_src, partition_map))
                
                unique_dst = block.dstnodes()
                dst_node_cnts.append(get_node_cnts(unique_dst, partition_map))
                
                # edge_id = block.edata["_ID"]
                
            log_step(rank, epoch, step, step_per_epoch, timer)

    src_node_cnts = lst_to_tensor(src_node_cnts)
    dst_node_cnts = lst_to_tensor(dst_node_cnts)
    input_node_cnts = lst_to_tensor(input_node_cnts)
    loc_edge_cnts = lst_to_tensor(loc_edge_cnts)
    crs_edge_cnts = lst_to_tensor(crs_edge_cnts)
    
    all_src = torch.zeros((src_node_cnts.shape[0] * config.world_size, config.world_size), device=rank, dtype=src_node_cnts.dtype)
    all_dst = torch.zeros((src_node_cnts.shape[0] * config.world_size, config.world_size), device=rank, dtype=src_node_cnts.dtype)
    all_loc = torch.zeros((src_node_cnts.shape[0] * config.world_size, config.world_size), device=rank, dtype=src_node_cnts.dtype)
    all_crs = torch.zeros((src_node_cnts.shape[0] * config.world_size, config.world_size), device=rank, dtype=src_node_cnts.dtype)
    all_input = torch.zeros((input_node_cnts.shape[0] * config.world_size, config.world_size), device=rank, dtype=input_node_cnts.dtype)
    
    dist.all_gather_into_tensor(all_src, src_node_cnts)
    dist.all_gather_into_tensor(all_dst, dst_node_cnts)
    dist.all_gather_into_tensor(all_input, input_node_cnts)
    dist.all_gather_into_tensor(all_loc, loc_edge_cnts)
    dist.all_gather_into_tensor(all_crs, crs_edge_cnts)
    
    if rank == 0:
        all_src = all_src.to("cpu")
        all_dst = all_dst.to("cpu")
        all_loc = all_loc.to("cpu")
        all_crs = all_crs.to("cpu")
        all_input = all_input.to("cpu")

        out_dir = os.path.join(config.data_dir, "workload", config.graph_name)
        print("saving to", out_dir)
        os.makedirs(out_dir, exist_ok=True)
                
        state = {
            "src": all_src,
            "dst": all_dst,
            "input": all_input,
            "loc": all_loc,
            "crs": all_crs
        }
        
        file_name = f"{config.graph_name}_w{config.world_size}_{config.partition_type}.pt"
        torch.save(state, os.path.join(out_dir, file_name))
    ddp_exit()
