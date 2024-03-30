import torch
import os
import csv
import torch.distributed as dist
from .config import Config
from .profiler import Profiler

def ddp_setup(local_rank, local_world_size, node_rank, num_nodes):
    if num_nodes == 0:
        print("local")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    else:
        print("distributed")
        os.environ["MASTER_ADDR"] = "3.139.229.20"
        os.environ["MASTER_PORT"] = "12355"

    global_rank = node_rank * local_world_size + local_rank
    global_world_size = local_world_size * num_nodes
    print(global_rank, global_world_size)
    dist.init_process_group(backend="nccl", rank = global_rank, world_size=global_world_size)
    torch.cuda.set_device(local_rank)

def ddp_exit():
    dist.destroy_process_group()

def write_to_csv(out_path, configs: list[Config], profilers: list[Profiler]):
    assert(len(configs) == len(profilers))
    def get_row(header, content):
        res = {}
        for k, v in zip(header, content):
            res[k] = v
        return res
    
    has_header = os.path.isfile(out_path)
    with open(out_path, 'a') as f:
        header = configs[0].header() + profilers[0].header()
        writer = csv.DictWriter(f, fieldnames=header)        
        if not has_header:
            writer.writeheader()
        for config, profiler in zip(configs, profilers):
            profiler.set_epoch_num(config.num_epoch)
            row = get_row(config.header() + profiler.header(), config.content() + profiler.content())
            writer.writerow(row)
    print("Experiment result has been written to: ", out_path)
