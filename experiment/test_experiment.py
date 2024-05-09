from torch.multiprocessing import spawn
import torch.multiprocessing as mp

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def bandwidth_experiment(rank, recv_gpu_id):
    world_size = 2
    ddp_setup(rank,world_size)

    if rank == 1:
        gpu_id = recv_gpu_id
    if rank == 0:
        gpu_id = 0
    torch.cuda.set_device(gpu_id)
    data = torch.ones(10 * 1024 * 1024 * 1024 // 4, dtype = torch.int32)
    for chunk_size in # 10KB to 10GB:
        for _ in rank
            e1 = torch.cuda.Event(enable_timing = False)
            for j in range(n_runs):



if __name__ == "__main__":
    # rank 0 to rank 1

    for recv_gpu in range(1):
        mp.spawn(bandwidth_experiment, args = recv_gpu, n_procs = 2)
