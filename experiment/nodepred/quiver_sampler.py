import torch, quiver, dgl
from dgl import create_block
from torch.utils.data import DataLoader

def partition_ids(rank: int, world_size: int, target_idx: torch.Tensor) -> torch.Tensor:
    step = int(target_idx.shape[0] / world_size)
    start_idx = rank * step
    end_idx = start_idx + step
    loc_ids = target_idx[start_idx : end_idx]
    return loc_ids.to(rank)

class QuiverGraphSageSampler():
    def __init__(self, sampler: quiver.pyg.GraphSageSampler):
        self.sampler = sampler
    
    def sample_dgl(self, seeds):
        """Sample k-hop neighbors from input_nodes

        Args:
            input_nodes (torch.LongTensor): seed nodes ids to sample from
        Returns:
            Tuple: Return results are the same with Dgl's sampler
            1. input_ndoes # to extract features
            2. output_nodes # to prefict label
            3. blocks # dgl blocks
        """
        self.sampler.lazy_init_quiver()
        blocks = []
        nodes = seeds

        for size in self.sampler.sizes:
            out, cnt = self.sampler.sample_layer(nodes, size)
            frontier, row_idx, col_idx = self.sampler.reindex(nodes, out, cnt)
            block = create_block(('coo', (col_idx, row_idx)), num_dst_nodes=nodes.shape[0], num_src_nodes=frontier.shape[0], device=self.sampler.device)
            blocks.insert(0, block)
            nodes = frontier
        return nodes, seeds, blocks
    
class QuiverDglSageSample():
    def __init__(self, 
                 rank: int,
                 world_size: int,
                 batch_size: int, 
                 target_idx:torch.Tensor, 
                 sampler: quiver.pyg.GraphSageSampler):
        self.rank = rank
        self.target_idx = partition_ids(rank, world_size, target_idx)
        self.batch_size = batch_size // world_size
        self.idx_loader = DataLoader(self.target_idx, batch_size=self.batch_size)
        self.iter = iter(self.idx_loader)
        self.sampler = QuiverGraphSageSampler(sampler)
    
    def reset(self):
        self.idx_loader = DataLoader(self.target_idx, batch_size=self.batch_size)
        self.iter = iter(self.idx_loader)
        
    def __iter__(self):
        self.iter = iter(self.idx_loader)
        return self

    def __next__(self):
        seeds = next(self.iter)
        return self.sampler.sample_dgl(seeds)