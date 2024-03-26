import os
import torch.distributed as dist
import torch

def ddp_setup( rank, world_size):
    os.environ["MASTER_ADDR"] = "3.139.229.20"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = 0
    torch.cuda.set_device(device)

def ddp_exit():
    dist.destroy_process_group()

def test_distributed(node):
    ddp_setup(node, world_size=2)
    a = torch.ones(100,100).to(0)
    if node == 0:
        dist.send(a * 2, 1)
    if node == 1:
        dist.recv(a, 0)
        print("recieved", a)
    ddp_exit()


if __name__ == "__main__":
    node = 0
    test_distributed(node)
