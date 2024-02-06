import dgl.backend as F
from torch import Tensor
from ..heterograph import DGLBlock
from .._ffi.function import _init_api
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

def UseBitmap(use_bitmap: bool) -> None:
    return _CAPI_UseBitmap(use_bitmap)

# def SampleBatches(seeds: Tensor, batch_len: int, replace: bool, batch_layer = 2):
#     return _CAPI_SampleBatches(F.to_dgl_nd(seeds), batch_len, replace, batch_layer)

def GetBlockData(batch_id: int, layer: int):
    data = _CAPI_GetBlockData(batch_id, layer)
    return F.from_dgl_nd(data)

def GetBlocks(batch_id: int, reindex = True, layers: int = 3) -> (Tensor, Tensor, list[DGLBlock]):
    input_node = _CAPI_GetInputNode(batch_id)
    input_node = F.from_dgl_nd(input_node)
    output_node = _CAPI_GetOutputNode(batch_id)
    output_node = F.from_dgl_nd(output_node)
    blocks = []
    for layer in range(layers):
        gidx = _CAPI_GetBlock(batch_id, layer, reindex)
        block = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
        block.edata["_ID"] = GetBlockData(batch_id, layer)
        blocks.insert(0, block)
    
    return input_node, output_node, blocks

def Increment(array: Tensor, row: Tensor):
    return _CAPI_Increment(F.to_dgl_nd(array), F.to_dgl_nd(row))

def COO2CSR(v_num: int, src: Tensor, dst: Tensor, data: Tensor=Tensor([]), to_undirected : bool = True):
    src = F.to_dgl_nd(src)
    dst = F.to_dgl_nd(dst)
    data = F.to_dgl_nd(data)
    ret = _CAPI_COO2CSR(v_num, src, dst, data, to_undirected)
    
    indptr = F.from_dgl_nd(ret(0))
    indices = F.from_dgl_nd(ret(1))
    data = F.from_dgl_nd(ret(2))
    return (indptr, indices, data)

def MakeSym(indptr:Tensor, indices:Tensor, data: Tensor=Tensor([])):
    indptr = F.to_dgl_nd(indptr)
    indices = F.to_dgl_nd(indices)
    data = F.to_dgl_nd(data)
    ret = _CAPI_MakeSym(indptr, indices, data)
    
    indptr = F.from_dgl_nd(ret(0))
    indices = F.from_dgl_nd(ret(1))
    data = F.from_dgl_nd(ret(2))
    return (indptr, indices, data)

def ReindexCSR(indptr: Tensor, indices: Tensor):
    indptr = F.to_dgl_nd(indptr)
    indices = F.to_dgl_nd(indices)
    ret = _CAPI_ReindexCSR(indptr, indices)
    indptr = F.from_dgl_nd(ret(0))
    indices = F.from_dgl_nd(ret(1))
    return (indptr, indices)

# compact indptr to remove the edges not in the flag
def CompactCSR(indptr: Tensor, flag: Tensor):
    indptr = F.to_dgl_nd(indptr)
    flag = F.to_dgl_nd(flag)
    _indptr = _CAPI_CompactCSR(indptr, flag)
    return F.from_dgl_nd(_indptr)

def LoadSNAP(in_file: str, to_sym=True):
    ret = _CAPI_LoadSNAP(in_file, to_sym)
    indptr = F.from_dgl_nd(ret(0))
    indices = F.from_dgl_nd(ret(1))
    return indptr, indices

def ExpandIndptr(indptr: Tensor):
    src = _CAPI_ExpandIndptr(F.to_dgl_nd(indptr))
    return F.from_dgl_nd(src)