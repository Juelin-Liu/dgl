from .._ffi.function import _init_api
import dgl.backend as F
from torch import Tensor
from ..heterograph import DGLBlock

_init_api("dgl.dev", __name__)

def SetGraph(indptr: Tensor, indices: Tensor, data: Tensor = None) -> None:
    if data == None:
        data = Tensor([])
    _CAPI_SetGraph(F.zerocopy_to_dgl_ndarray(indptr), 
                   F.zerocopy_to_dgl_ndarray(indices), 
                   F.zerocopy_to_dgl_ndarray(data))

def SetFanout(fanout: list[int]) -> None:
    _CAPI_SetFanout(fanout)

def SampleBatch(seeds: Tensor, replace : bool = False) -> int:
    return _CAPI_SampleBatch(F.zerocopy_to_dgl_ndarray(seeds), replace)
    
def SampleBatches(seeds: Tensor, batch_len: int, replace: bool, batch_layer = 2):
    return _CAPI_SampleBatches(F.zerocopy_to_dgl_ndarray(seeds), batch_len, replace, batch_layer)

def GetBlocks(batch_id: int, reindex = True, layers: int = 3)->(Tensor, Tensor, list[DGLBlock]):
    input_node = _CAPI_GetInputNode(batch_id)
    input_node = F.zerocopy_from_dgl_ndarray(input_node)
    output_node = _CAPI_GetOutputNode(batch_id)
    output_node = F.zerocopy_from_dgl_ndarray(output_node)
    blocks = []
    for layer in range(layers):
        gidx = _CAPI_GetBlock(batch_id, layer, reindex)
        block = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
        blocks.insert(0, block)
    
    return input_node, output_node, blocks

def GetBlockData(batch_id: int, layer: int):
    data = _CAPI_GetBlockData(batch_id, layer)
    return F.zerocopy_from_dgl_ndarray(data)
