from .util import *
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.cuda import device_count
from .. import DGLGraph
@dataclass
class SampleConfig:
    rank: int = 0
    world_size: int = 1
    batch_size:int = 1024
    replace:bool = False
    mode: str = "uva" # must be one of ["uva", "gpu"]
    reindex: bool = True # wheter to make the sampled vertex to be 0 indexed
    fanouts: list[int] = None
    def header(self):
        return ["rank","world_size", "batch_size", "replace", "mode", "reindex", "fanouts"]
    def content(self):
        return [self.rank,
                self.world_size,
                self.batch_size,
                self.replace,
                self.mode,
                self.reindex,
                self.fanouts
                ]
        
class GraphDataloader:
    def __init__(self, g: DGLGraph, train_idx: Tensor, config: SampleConfig):
        assert(config.mode in ["uva", "gpu"])
        assert(config.fanouts is not None)
        assert(config.rank < device_count())
        assert("csc" in g.formats()["created"])
        self.device = f"cuda:{config.rank}"
        self.idx_loader = DataLoader(train_idx.to(self.device), batch_size=config.batch_size)
        self.iter = iter(self.idx_loader)
        self.config = config
        if config.mode == "uva":
            self.g = g.pin_memory_()
        elif config.mode == "gpu":
            self.g = g.to(self.device)
        
        indptr, indices, edges = self.g.adj_tensors("csc")
        SetGraph(indptr, indices)
        SetFanout(config.fanouts)

    def __iter__(self):
        self.iter = iter(self.idx_loader)
        return self
    
    def __next__(self) -> (Tensor, Tensor, list[DGLBlock]):
        seeds = next(self.iter)
        batch_id = SampleBatch(seeds, self.config.replace)
        return GetBlocks(batch_id, reindex=self.config.reindex, layers=len(self.config.fanouts))
    