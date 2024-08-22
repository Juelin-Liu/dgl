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
    NDArray _input_nodes; // hold the input_nodes until next prefetch is evoked
    QueryIdx _idx; // hold the query idx until next prefetch is evoked
    std::shared_ptr<DeviceBitmap> _cached_bitmap{nullptr};
    bool init{false};
    bool cached{false};
    int64_t _feat_width{-1};
    DGLContext _ctx{};
    cudaEvent_t _event{nullptr};
    cudaStream_t _prefetch_stream{nullptr};
    void Init(DGLContext ctx, const NDArray& pinned_feat, const NDArray& cache_ids);

   public:
    FeatCache(DGLContext ctx, const NDArray& pinned_feat, const NDArray& cache_ids);

    ~FeatCache() = default;

    NDArray Prefetch(const NDArray& input_nodes, cudaEvent_t event_to_wait = nullptr);

    void Sync() const;
  };
}
#endif  // DGL_FEAT_CACHE_CUH