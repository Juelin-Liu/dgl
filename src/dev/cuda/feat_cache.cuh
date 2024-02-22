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
  class FeatCache {
   private:
    NDArray _cached_feat;
    NDArray _pinned_feat;
    NDArray _label;
    DeviceBitmap _cached_bitmap;
    bool init{false};
    bool cached{false};
    int64_t next_id{0};
    DGLContext _ctx{};
    cudaEvent_t _event{nullptr};
    cudaStream_t _prefetch_stream{nullptr};

   public:
    FeatCache() = default;
    ~FeatCache() = default;

    void Init(DGLContext ctx, const NDArray& pinned_feat, const NDArray& cache_ids);

    NDArray Prefetch(const NDArray& input_nodes, cudaEvent_t event_to_wait = nullptr);

    void Sync() const;
  };
}
#endif  // DGL_FEAT_CACHE_CUH
