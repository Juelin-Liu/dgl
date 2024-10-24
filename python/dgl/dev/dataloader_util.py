from dgl.backend import to_dgl_nd, from_dgl_nd
from torch import Tensor
from ..heterograph import DGLBlock
from .._ffi.function import _init_api
_init_api("dgl.dev", __name__)
from torch.cuda import nvtx

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
#
def UseBitmap(use_bitmap: bool) -> None:
    return _CAPI_UseBitmap(use_bitmap)

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
    return input_node, output_node, blocks