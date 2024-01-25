import torchmetrics.functional as MF
import torch.distributed as dist
import gc
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from node.utils import *
from dgl.dev import *

def freq(config: Config):
    graph, train_idx, valid_idx, test_idx = load_topo(config, is_pinned=False)
    graph: dgl.DGLGraph = dgl.remove_self_loop(graph)
    graph.pin_memory_()
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
    step_per_epoch = dataloader.target_idx.shape[0] // dataloader.batch_size + 1
    
    print(f"sampling on device: {device}")        
    timer = Timer()
    epoch = config.num_epoch
    
    v_num = graph.num_nodes()
    e_num = graph.num_edges()
    
    input_node_weight = torch.zeros((v_num,),dtype=torch.int32, device=rank)
    src_node_weight = torch.zeros((v_num,),dtype=torch.int32, device=rank)
    dst_node_weight = torch.zeros((v_num,),dtype=torch.int32, device=rank)
    edge_weight = torch.zeros((e_num,), dtype=torch.int32, device=rank) # TODO enable int16
    
    for epoch in range(config.num_epoch):
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
        input_node_weight = input_node_weight.to("cpu") // config.num_epoch
        src_node_weight = src_node_weight.to("cpu") // config.num_epoch
        dst_node_weight = dst_node_weight.to("cpu") // config.num_epoch
        edge_weight = edge_weight.to("cpu") // config.num_epoch
        print("saving to", out_dir)
        torch.save(input_node_weight, f"{out_dir}/input_node_weight.pt")
        torch.save(src_node_weight, f"{out_dir}/src_node_weight.pt")
        torch.save(dst_node_weight, f"{out_dir}/dst_node_weight.pt")
        torch.save(edge_weight, f"{out_dir}/edge_weight.pt")
    ddp_exit()