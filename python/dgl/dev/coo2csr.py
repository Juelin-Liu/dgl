from dgl.backend import to_dgl_nd, from_dgl_nd
from torch import Tensor
from ..heterograph import DGLBlock
from .._ffi.function import _init_api

_init_api("dgl.dev", __name__)

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

def Increment(array: Tensor, row: Tensor):
    return _CAPI_Increment(to_dgl_nd(array), to_dgl_nd(row))