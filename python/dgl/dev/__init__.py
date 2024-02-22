# from .dataloader import GraphDataloader
# from .splitloader import SplitGraphLoader
from .._ffi.function import _init_api

_init_api("dgl.dev", __name__)

def CudaProfilerStart():
    _CAPI_cudaProfilerStart()
    
def CudaProfilerStop():
    _CAPI_cudaProfilerStop()