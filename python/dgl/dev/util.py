from dgl.backend import to_dgl_nd, from_dgl_nd
from torch import Tensor
from ..heterograph import DGLBlock
from .._ffi.function import _init_api
_init_api("dgl.dev", __name__)
import torch.cuda.nvtx as nvtx

def SetGraph(indptr: Tensor, indices: Tensor, data: Tensor = None) -> None:
    if data == None:
        data = Tensor([])
    _CAPI_SetGraph(to_dgl_nd(indptr), 
                   to_dgl_nd(indices), 
                   to_dgl_nd(data))

def SetFanout(fanout: list[int]) -> None:
    _CAPI_SetFanout(fanout)

def SampleBatch(seeds: Tensor, replace : bool = False) -> int:
    return _CAPI_SampleBatch(to_dgl_nd(seeds), replace)

def UseBitmap(use_bitmap: bool) -> None:
    return _CAPI_UseBitmap(use_bitmap)

# def SampleBatches(seeds: Tensor, batch_len: int, replace: bool, batch_layer = 2):
#     return _CAPI_SampleBatches(to_dgl_nd(seeds), batch_len, replace, batch_layer)

def GetBlockData(batch_id: int, layer: int):
    data = _CAPI_GetBlockData(batch_id, layer)
    return from_dgl_nd(data)

def GetBlocks(batch_id: int, reindex: bool = True, layers: int = 3, edge_data: bool = True) -> (Tensor, Tensor, list[DGLBlock]):
    input_node = _CAPI_GetInputNode(batch_id)
    input_node = from_dgl_nd(input_node)
    output_node = _CAPI_GetOutputNode(batch_id)
    output_node = from_dgl_nd(output_node)
    blocks = []
    for layer in range(layers):
        nvtx.range_push("capi get block")

        gidx = _CAPI_GetBlock(batch_id, layer, reindex)

        nvtx.range_pop()
        nvtx.range_push("dgl create block")

        block = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
        nvtx.range_pop()
        
        nvtx.range_push("dgl get block data")
        if edge_data:
            block.edata["_ID"] = GetBlockData(batch_id, layer)
        nvtx.range_pop()
        blocks.insert(0, block)
    # avoid python for loop overhead
    # if layers == 4:
    #     blocks = [None, None, None, None]
    #     blocks[0] = DGLBlock(_CAPI_GetBlock(batch_id, layers - 0 - 1, reindex), (['_N'], ['_N']), ['_E'])
    #     blocks[1] = DGLBlock(_CAPI_GetBlock(batch_id, layers - 1 - 1, reindex), (['_N'], ['_N']), ['_E'])
    #     blocks[2] = DGLBlock(_CAPI_GetBlock(batch_id, layers - 2 - 1, reindex), (['_N'], ['_N']), ['_E'])
    #     blocks[3] = DGLBlock(_CAPI_GetBlock(batch_id, layers - 3 - 1, reindex), (['_N'], ['_N']), ['_E'])
    #
    #     blocks[0].edata["_ID"] = GetBlockData(batch_id, layers - 0 - 1)
    #     blocks[1].edata["_ID"] = GetBlockData(batch_id, layers - 1 - 1)
    #     blocks[2].edata["_ID"] = GetBlockData(batch_id, layers - 2 - 1)
    #     blocks[3].edata["_ID"] = GetBlockData(batch_id, layers - 3 - 1)
    #
    # elif layers == 3:
    #     nvtx.range_push("create blocks")
    #     blocks = [None, None, None]
    #     blocks[0] = DGLBlock(_CAPI_GetBlock(batch_id, layers - 0 - 1, reindex), (['_N'], ['_N']), ['_E'])
    #     blocks[1] = DGLBlock(_CAPI_GetBlock(batch_id, layers - 1 - 1, reindex), (['_N'], ['_N']), ['_E'])
    #     blocks[2] = DGLBlock(_CAPI_GetBlock(batch_id, layers - 2 - 1, reindex), (['_N'], ['_N']), ['_E'])
    #     nvtx.range_pop()
    #
    #     nvtx.range_push("get block data")
    #     if edge_data:
    #         blocks[0].edata["_ID"] = GetBlockData(batch_id, layers - 0 - 1)
    #         blocks[1].edata["_ID"] = GetBlockData(batch_id, layers - 1 - 1)
    #         blocks[2].edata["_ID"] = GetBlockData(batch_id, layers - 2 - 1)
    #     nvtx.range_pop()
    #
    # elif layers == 2:
    #     blocks = [None, None]
    #     blocks[0] = DGLBlock(_CAPI_GetBlock(batch_id, layers - 0 - 1, reindex), (['_N'], ['_N']), ['_E'])
    #     blocks[1] = DGLBlock(_CAPI_GetBlock(batch_id, layers - 1 - 1, reindex), (['_N'], ['_N']), ['_E'])
    #
    #     blocks[0].edata["_ID"] = GetBlockData(batch_id, layers - 0 - 1)
    #     blocks[1].edata["_ID"] = GetBlockData(batch_id, layers - 1 - 1)
    #
    # else:
    #     blocks = []
    #     for layer in range(layers):
    #         nvtx.range_push("capi get block")
    #
    #         gidx = _CAPI_GetBlock(batch_id, layer, reindex)
    #
    #         nvtx.range_pop()
    #         nvtx.range_push("dgl create block")
    #
    #         block = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
    #         nvtx.range_pop()
    #
    #         block.edata["_ID"] = GetBlockData(batch_id, layer)
    #         blocks[layers - layer - 1] = block
    
    return input_node, output_node, blocks

def Increment(array: Tensor, row: Tensor):
    return _CAPI_Increment(to_dgl_nd(array), to_dgl_nd(row))

def COO2CSR(v_num: int, src: Tensor, dst: Tensor, data: Tensor=Tensor([]), to_undirected : bool = True):
    src = to_dgl_nd(src)
    dst = to_dgl_nd(dst)
    data = to_dgl_nd(data)
    ret = _CAPI_COO2CSR(v_num, src, dst, data, to_undirected)
    
    indptr = from_dgl_nd(ret(0))
    indices = from_dgl_nd(ret(1))
    data = from_dgl_nd(ret(2))
    return (indptr, indices, data)

def MakeSym(indptr:Tensor, indices:Tensor, data: Tensor=Tensor([])):
    indptr = to_dgl_nd(indptr)
    indices = to_dgl_nd(indices)
    data = to_dgl_nd(data)
    ret = _CAPI_MakeSym(indptr, indices, data)
    
    indptr = from_dgl_nd(ret(0))
    indices = from_dgl_nd(ret(1))
    data = from_dgl_nd(ret(2))
    return (indptr, indices, data)

def ReindexCSR(indptr: Tensor, indices: Tensor):
    indptr = to_dgl_nd(indptr)
    indices = to_dgl_nd(indices)
    ret = _CAPI_ReindexCSR(indptr, indices)
    indptr = from_dgl_nd(ret(0))
    indices = from_dgl_nd(ret(1))
    return (indptr, indices)

# compact indptr to remove the edges not in the flag
def CompactCSR(indptr: Tensor, flag: Tensor):
    indptr = to_dgl_nd(indptr)
    flag = to_dgl_nd(flag)
    _indptr = _CAPI_CompactCSR(indptr, flag)
    return from_dgl_nd(_indptr)

def LoadSNAP(in_file: str, to_sym=True):
    ret = _CAPI_LoadSNAP(in_file, to_sym)
    indptr = from_dgl_nd(ret(0))
    indices = from_dgl_nd(ret(1))
    return indptr, indices

def ExpandIndptr(indptr: Tensor):
    src = _CAPI_ExpandIndptr(to_dgl_nd(indptr))
    return from_dgl_nd(src)
