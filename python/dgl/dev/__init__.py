from .._ffi.function import _init_api
import dgl.backend as F
from torch import Tensor
from ..heterograph import DGLBlock

_init_api("dgl.dev", __name__)

def SetGraph(indptr: Tensor, indices: Tensor, data: Tensor = None) -> None:
    if data == None:
        data = Tensor([])
    _CAPI_SetGraph(F.to_dgl_nd(indptr), 
                   F.to_dgl_nd(indices), 
                   F.to_dgl_nd(data))

def SetFanout(fanout: list[int]) -> None:
    _CAPI_SetFanout(fanout)

def SampleBatch(seeds: Tensor, replace : bool = False) -> int:
    return _CAPI_SampleBatch(F.to_dgl_nd(seeds), replace)
    
def SampleBatches(seeds: Tensor, batch_len: int, replace: bool, batch_layer = 2):
    return _CAPI_SampleBatches(F.to_dgl_nd(seeds), batch_len, replace, batch_layer)

def GetBlocks(batch_id: int, reindex = True, layers: int = 3):
    input_node = _CAPI_GetInputNode(batch_id)
    input_node = F.from_dgl_nd(input_node)
    output_node = _CAPI_GetOutputNode(batch_id)
    output_node = F.from_dgl_nd(output_node)
    blocks = []
    for layer in range(layers):
        gidx = _CAPI_GetBlock(batch_id, layer, reindex)
        block = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
        blocks.insert(0, block)
    
    return input_node, output_node, blocks

def GetBlockData(batch_id: int, layer: int):
    data = _CAPI_GetBlockData(batch_id, layer)
    return F.from_dgl_nd(data)

def Increment(array: Tensor, row: Tensor):
    return _CAPI_Increment(F.to_dgl_nd(array), F.to_dgl_nd(row))

def COO2CSR(v_num: int, src: Tensor, dst: Tensor, data: Tensor):
    src = F.to_dgl_nd(src)
    dst = F.to_dgl_nd(dst)
    data = F.to_dgl_nd(data)
    ret = _CAPI_COO2CSR(v_num, src, dst, data)
    degree = F.from_dgl_nd(ret(0))
    indptr = F.from_dgl_nd(ret(1))
    indices = F.from_dgl_nd(ret(2))
    data = F.from_dgl_nd(ret(3))
    return (degree, indptr, indices, data)