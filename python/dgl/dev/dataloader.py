from .dataloader_util import *
from .util import SampleConfig, IdxLoader
from torch.cuda import device_count, nvtx
from torch import int32
from .. import DGLGraph

class GraphDataloader:
    def __init__(self, g: DGLGraph, target_idx: Tensor, config: SampleConfig):
        assert (config.mode in ["uva", "gpu"])
        assert (config.fanouts is not None)
        assert (config.rank < device_count())
        assert ("csc" in g.formats()["created"])
        self.device = f"cuda:{config.rank}"
        self.rank = config.rank
        self.fanouts = config.fanouts
        self.replace = config.replace
        self.reindex = config.reindex
        self.batch_size = config.batch_size // config.world_size
        self.config = config

        if config.mode == "uva":
            self.g = g.pin_memory_()
        elif config.mode == "gpu":
            print("copying graph to:", self.device)
            if g.num_edges() < 2**31:
                self.g = g.int().to(self.device)
                target_idx = target_idx.type(int32)
            else:
                self.g = g.to(self.device)
        
        self.target_type = target_idx.dtype
        
        indptr, indices, _ = self.g.adj_tensors("csc")
        SetGraph(indptr, indices)
        SetFanout(config.fanouts)
        self.set_target_idx(target_idx)

    def set_fanout(self, fanouts):
        self.fanouts = fanouts
        SetFanout(fanouts)
        
    def set_target_idx(self, target_idx):
        self.loc_idx_size = target_idx.shape[0] // self.config.world_size + 1
        # self.global_target_idx = target_idx
        self.local_target_idx = target_idx[self.rank * self.loc_idx_size:(self.rank + 1) * self.loc_idx_size].type(self.target_type).clone()
        self.shuffle = True
        self.max_step_per_epoch = self.local_target_idx.shape[0] // self.batch_size
        self.idx_loader = IdxLoader(d=self.device, target_idx=self.local_target_idx,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     max_step_per_epoch=self.max_step_per_epoch)
        self.iter = iter(self.idx_loader)
        
    def reset(self):
        self.idx_loader = IdxLoader(d=self.device, target_idx=self.local_target_idx, 
                                    batch_size=self.batch_size, 
                                    shuffle=True, 
                                    max_step_per_epoch=self.max_step_per_epoch)
        self.iter = iter(self.idx_loader)

    def __iter__(self):
        self.iter = iter(self.idx_loader)
        return self

    def __next__(self) -> (Tensor, Tensor, list[DGLBlock]):
        seeds = next(self.iter)

        nvtx.range_push("start sampling")
        batch_id = SampleBatch(seeds, self.replace)
        nvtx.range_pop()

        nvtx.range_push("create block")
        ret = GetBlocks(batch_id, reindex=self.reindex, layers=len(self.fanouts))
        nvtx.range_pop()
        return ret
