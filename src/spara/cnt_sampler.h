//
// Created by juelin on 6/16/24.
//

#ifndef DGL_CNT_SAMPLER_H
#define DGL_CNT_SAMPLER_H

#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/container.h>

#include <memory>
#include <queue>
#include <utility>

#include "../graph/unit_graph.h"
#include "../runtime/cuda/cuda_common.h"
#include "../runtime/cuda/cuda_hashtable.cuh"
#include "cuda/map_edges.cuh"
#include "cuda/array_ops.cuh"

namespace dgl::dev {

class CntSampler {
 private:
  aten::CSRMatrix _csc;
  std::vector<int64_t> _fanouts;
  DGLContext _ctx{}; // inferred from the ctx of the seeds
  NDArray v_freq, e_freq;
  void *v_freq_ptr{nullptr};
  void *e_freq_ptr{nullptr};

  int64_t next_id{0};
  int64_t rank{0};
 public:
  static std::shared_ptr<CntSampler> Global() {
    static auto single_instance = std::make_shared<CntSampler>();
    return single_instance;
  }
  ~CntSampler() {
    if (v_freq_ptr) cudaFree(v_freq_ptr);
    if (e_freq_ptr) cudaFree(e_freq_ptr);
  }
  void setGraph(int32_t device_id, const NDArray& indptr, const NDArray& indices) {
    CHECK(indptr.IsPinned() || indptr->ctx.device_type != kDGLCPU)
        << "Indptr must be pinned or on gpu";
    CHECK(indices.IsPinned() || indices->ctx.device_type != kDGLCPU)
        << "Indices must be pinned or on gpu";

    int64_t nrows = indptr.NumElements() - 1;
    int64_t ncols = indices.NumElements();
    _csc = aten::CSRMatrix(nrows, ncols, indptr, indices);
    _ctx = DGLContext {kDGLCUDA, device_id};
    cudaSetDevice(device_id);

    LOG(INFO) << "Initializing v_freq and e_freq in ctx: " << _ctx;

    cudaMalloc(&v_freq_ptr, nrows * sizeof(int32_t));
    cudaMalloc(&e_freq_ptr, ncols * sizeof(int32_t));
    cudaMemset(v_freq_ptr, 0, nrows * sizeof(int32_t ));
    cudaMemset(e_freq_ptr, 0, ncols * sizeof(int32_t ));
    v_freq = NDArray::CreateFromRaw({nrows}, DGLDataType{kDGLInt, 32, 1}, _ctx, v_freq_ptr, false);
    e_freq = NDArray::CreateFromRaw({ncols}, DGLDataType{kDGLInt, 32, 1}, _ctx, e_freq_ptr, false);
  }

  void setFanouts(std::vector<int64_t> fanouts) {
    _fanouts = std::move(fanouts);
  }

  /*
   * seeds: the root vertex to be sampled from the graph
   * replace: use replace sampling or not
   */
  int64_t sampleOneBatch(const NDArray& seeds, bool replace) {
    CHECK(seeds.IsPinned() || seeds->ctx.device_type != DGLDeviceType::kDGLCPU)
        << "Seeds must be pinned or in GPU memory";

    auto frontier = seeds;
    for (size_t layer = 0; layer < _fanouts.size(); layer++) {
      int64_t fanout = _fanouts.at(layer);
      aten::COOMatrix block = aten::CSRRowWiseSampling(
          _csc, frontier, fanout, aten::NullArray(), replace);

      ATEN_ID_TYPE_SWITCH(block.col->dtype, IdType, {
        Increment<kDGLCUDA, int32_t, IdType>(v_freq, block.col);
        Increment<kDGLCUDA, int32_t, IdType>(e_freq, block.data);
      });

      if (layer != _fanouts.size() - 1) {
        frontier = getUnique({frontier, block.col});
      }
    }
    return next_id++;
  }

  NDArray getNodeFreq() {
    return v_freq;
  }

  NDArray getEdgeFreq() {
    return e_freq;
  }
};

}  // namespace dgl::dev

#endif  // DGL_CNT_SAMPLER_H
