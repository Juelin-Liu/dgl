import torch
import os
import csv
import torch.distributed as dist
from .config import Config
from .profiler import Profiler

def get_tensor_size(intensor: torch.Tensor):
    num_bytes = intensor.element_size() * intensor.nelement()
    return f"{round(num_bytes / 1e9, 1)}GB"

def ddp_setup(rank, world_size, backend="nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

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