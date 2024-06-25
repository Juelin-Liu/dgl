from dgl.backend import to_dgl_nd, from_dgl_nd
from torch import Tensor
from ..heterograph import DGLBlock
from .util import SampleConfig, IdxLoader
from .. import DGLGraph
from .._ffi.function import _init_api
from torch.cuda import device_count, nvtx
from torch import int32

_init_api("dgl.dev", __name__)

def CntSetGraph(rank: int, indptr: Tensor, indices: Tensor) -> None:
    _CAPI_Cnt_SetGraph(rank, to_dgl_nd(indptr),
                   to_dgl_nd(indices))

def CntSetFanout(fanout: list[int]) -> None:
    _CAPI_Cnt_SetFanout(fanout)

def CntSampleBatch(seeds: Tensor, replace : bool = False) -> int:
    return _CAPI_Cnt_SampleBatch(to_dgl_nd(seeds), replace)

def CntGetNodeFreq() -> Tensor:
    return from_dgl_nd(_CAPI_Cnt_GetNodeFreq())

def CntGetEdgeFreq() -> Tensor:
    return from_dgl_nd(_CAPI_Cnt_GetEdgeFreq())


class CntSampler:
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
        CntSetGraph(self.rank, indptr, indices)
        CntSetFanout(config.fanouts)
        self.set_target_idx(target_idx)

    def set_fanout(self, fanouts):
        self.fanouts = fanouts
        CntSetFanout(fanouts)
        
    def set_target_idx(self, target_idx):
        self.loc_idx_size = target_idx.shape[0] // self.config.world_size + 1
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

    def __next__(self) -> int:
        seeds = next(self.iter)
        batch_id = CntSampleBatch(seeds, self.replace)        
        return batch_id
