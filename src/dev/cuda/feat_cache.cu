//
// Created by juelin on 2/22/24.
//

#include "feat_cache.cuh"
#include "index_select.cuh"
#include "../../runtime/cuda/cuda_common.h"

namespace dgl::dev {
void FeatCache::Init(
    DGLContext ctx, const NDArray &pinned_feat, const NDArray &cache_ids) {
  _pinned_feat = pinned_feat;
  _ctx = ctx;
  CUDA_CALL(cudaEventCreate(&_event));
  CUDA_CALL(cudaStreamCreateWithFlags(&_prefetch_stream, cudaStreamNonBlocking));
  // copying features of cached ids to gpu
  if (cache_ids.NumElements() > 0) {
    CHECK_EQ(cache_ids->ctx, _ctx);
    auto stream = runtime::getCurrentCUDAStream();
    auto device = runtime::DeviceAPI::Get(_ctx);
    _cached_bitmap = std::make_shared<DeviceBitmap>(pinned_feat.NumElements(), _ctx, 32);
    ATEN_ID_TYPE_SWITCH(cache_ids->dtype, IdType, {
      _cached_bitmap->flag(cache_ids.Ptr<IdType>(), cache_ids.NumElements());
    });
    _cached_feat = IndexSelect(_pinned_feat, cache_ids, stream);
    device->StreamSync(_ctx, stream);
    cached = true;
  }
  init = true;
}

NDArray FeatCache::Prefetch(const dgl::runtime::NDArray &input_nodes, cudaEvent_t event_to_wait) {
  CHECK_EQ(cached, false) << "do not support caching for now";
  if (event_to_wait != nullptr) CUDA_CALL(cudaStreamWaitEvent(
      _prefetch_stream, event_to_wait));  // must wait until all events have finished
  if (cached) {
    // TODO: implement me
    return NDArray::Empty({0}, _pinned_feat->dtype, _ctx);
  } else {
    return IndexSelect(_pinned_feat, input_nodes, _prefetch_stream);
    CUDA_CALL(cudaEventRecord(_event, _prefetch_stream));
  }
}

void FeatCache::Sync() const {
  CUDA_CALL(cudaEventSynchronize(_event));
}
}  // namespace dgl::dev