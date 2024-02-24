//
// Created by juelin on 2/22/24.
//

#include "feat_cache.cuh"
#include "index_select.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include <nvtx3/nvtx3.hpp>
#include <dgl/aten/array_ops.h>

namespace dgl::dev {
FeatCache::FeatCache(DGLContext ctx, const NDArray& pinned_feat, const NDArray& cache_ids){
  Init(ctx, pinned_feat, cache_ids);
}

void FeatCache::Init(
    DGLContext ctx, const NDArray &pinned_feat, const NDArray &cache_ids) {
  _pinned_feat = pinned_feat;
  _ctx = ctx;
  _feat_width = _pinned_feat.NumElements() / _pinned_feat->shape[0];
  CUDA_CALL(cudaEventCreate(&_event));
  CUDA_CALL(cudaStreamCreateWithFlags(&_prefetch_stream, cudaStreamNonBlocking));
  // copying features of cached ids to gpu
  if (cache_ids.NumElements() > 0) {
    const auto [sorted_arr, sorted_idx] = aten::Sort(cache_ids, cache_ids->dtype.bits);
    auto d_sorted_cache_ids = sorted_arr.CopyTo(_ctx);
    auto stream = runtime::getCurrentCUDAStream();
    auto device = runtime::DeviceAPI::Get(_ctx);
    _cached_bitmap = std::make_shared<DeviceBitmap>(_pinned_feat->shape[0], _ctx, 1);

    ATEN_ID_TYPE_SWITCH(d_sorted_cache_ids->dtype, IdType, {
      _cached_bitmap->flag(
          d_sorted_cache_ids.Ptr<IdType>(), d_sorted_cache_ids.NumElements());
      _cached_bitmap->buildOffset();
    });

    _cached_feat = IndexSelect(_pinned_feat, d_sorted_cache_ids, stream);
    device->StreamSync(_ctx, stream);
    cached = true;
  }
  init = true;
}

NDArray FeatCache::Prefetch(const dgl::runtime::NDArray &input_nodes, cudaEvent_t event_to_wait) {
  if (event_to_wait != nullptr) CUDA_CALL(cudaStreamWaitEvent(_prefetch_stream, event_to_wait));

  NDArray ret;
  if (cached) {
    QueryIdx idx;
    ATEN_ID_TYPE_SWITCH(input_nodes->dtype, IdType, {
      idx = _cached_bitmap->queryBitmap(input_nodes.Ptr<IdType>(), input_nodes.NumElements());
    });
    CUDA_CALL(cudaEventRecord(_event, runtime::getCurrentCUDAStream())); // must wait until all events have finished in queryBitmap
    CUDA_CALL(cudaStreamWaitEvent(_prefetch_stream, _event));
    ret = NDArray::Empty({input_nodes->shape[0], _feat_width}, _pinned_feat->dtype, _ctx);
    IndexSelect(_pinned_feat, idx._missReadId, idx._missWriteIdx, ret, _prefetch_stream);
    IndexSelect(_cached_feat, idx._hitReadIdx, idx._hitWriteIdx, ret, _prefetch_stream);
  } else {
    CUDA_CALL(cudaEventRecord(_event, runtime::getCurrentCUDAStream())); // must wait until all events have finished in queryBitmap
    CUDA_CALL(cudaStreamWaitEvent(_prefetch_stream, _event));
    ret = IndexSelect(_pinned_feat, input_nodes, _prefetch_stream);
  }
  CUDA_CALL(cudaEventRecord(_event, _prefetch_stream));
  return ret;
}

void FeatCache::Sync() const {
  CUDA_CALL(cudaEventSynchronize(_event));
}
}  // namespace dgl::dev