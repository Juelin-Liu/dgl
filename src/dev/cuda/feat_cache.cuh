//
// Created by juelin on 2/22/24.
//

#ifndef DGL_FEAT_CACHE_CUH
#define DGL_FEAT_CACHE_CUH
#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include "bitmap.h"

namespace dgl::dev
{
  class GpuCache {
   private:
    NDArray _cached_feat;
    NDArray _pinned_feat;
    NDArray _label;
    DeviceBitmap _bitmap;
    bool inited{false};
    void *feat_buff;
    void *label_buff;

    int64_t batch_id{0};
    cudaEvent_t _event;
    cudaStream_t _prefetch_stream;

   public:
    GpuFeatCache() = default;

    void init(const NDArray& pinned_feat, const NDArray& cached_id);

    int64_t Prefetch(const NDArray& input_nodes, const NDArray& output_nodes);

    NDArray GetFeature(int64_t batch_id);

    NDArray GetLabel(int64_t batch_id);
  };
}
#endif  // DGL_FEAT_CACHE_CUH
