from dgl.backend import to_dgl_nd, from_dgl_nd
from torch import Tensor

from torch.cuda import nvtx, device_count
from .. import DGLGraph

from .util import IdxLoader, SampleConfig
from .splitloader_util import *
from .._ffi.function import _init_api
_init_api("dgl.dev", __name__)


class SplitGraphLoader:
    def __init__(self, g: DGLGraph, partition_map: Tensor, target_idx: Tensor, config: SampleConfig):
        assert (config.mode in ["uva", "gpu"])
        assert (config.fanouts is not None)
        assert (config.rank < device_count())
        assert ("csc" in g.formats()["created"])

        self.device = f"cuda:{config.rank}"
        self.rank = config.rank
        self.loc_idx_size = target_idx.shape[0] // config.world_size + 1
        self.fanouts = config.fanouts
        self.replace = config.replace
        self.reindex = config.reindex
        self.config = config
        self.partition_map = partition_map.to(self.rank)

        if config.mode == "uva":
            self.g = g.pin_memory_()
        elif config.mode == "gpu":
            self.g = g.to(self.device)

        indptr, indices, _ = self.g.adj_tensors("csc")
        SetGraph(indptr, indices)
        SetFanout(config.fanouts)
        SetPartitionMap(config.num_partition, self.partition_map)
        SetRank(config.rank, config.world_size)
        self.set_target_idx(target_idx)
        
    def set_fanout(self, fanouts):
        self.fanouts = fanouts
        SetFanout(fanouts)
        
    def set_target_idx(self, target_idx):
        target_idx = target_idx.to(self.rank)
        self.global_target_idx = target_idx
        self.target_idx = target_idx[self.partition_map[target_idx] == self.rank].to(self.rank)
        self.num_step_per_epoch = self.global_target_idx.shape[0] // self.config.batch_size
        self.batch_size = self.target_idx.shape[0] // self.num_step_per_epoch
        self.idx_loader = IdxLoader(target_idx=self.target_idx,
                                    batch_size=self.batch_size,
                                    max_step_per_epoch=self.num_step_per_epoch,
                                    shuffle=True)
        self.iter = iter(self.idx_loader)
        
    def reset(self):
        self.idx_loader = IdxLoader(self.target_idx, batch_size=self.batch_size)
        self.iter = iter(self.idx_loader)

    def __iter__(self):
        self.iter = iter(self.idx_loader)
        return self

    def __next__(self) -> (Tensor, Tensor, list[DGLBlock]):
        nvtx.range_push("start next")
        seeds = next(self.iter)
        nvtx.range_pop()

        nvtx.range_push("start sampling")
        batch_id = SampleBatch(seeds, self.replace)
        nvtx.range_pop()

        nvtx.range_push("create block")
        ret = GetBlocks(batch_id, reindex=self.reindex, layers=len(self.fanouts))
        nvtx.range_pop()
        return ret
