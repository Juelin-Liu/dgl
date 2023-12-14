//
// Created by juelinliu on 12/10/23.
//

#ifndef DGL_SAMPLER_H
#define DGL_SAMPLER_H
#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/container.h>

#include <queue>
#include <utility>

#include "../graph/unit_graph.h"
#include "../runtime/cuda/cuda_common.h"
#include "../runtime/cuda/cuda_hashtable.cuh"
#include "cuda/map_edges.cuh"

namespace dgl::dev {

struct GraphBatch {
  int64_t _batch_id{-1};
  bool reindexed{false};
  std::vector<aten::COOMatrix> _blocks;
  std::vector<NDArray> _frontiers;
  std::vector<HeteroGraphRef> _blockrefs;
  GraphBatch(
      int64_t batch_id, std::vector<NDArray> frontiers,
      std::vector<aten::COOMatrix> blocks)
      : _batch_id{batch_id},
        _blocks{std::move(blocks)},
        _frontiers{std::move(frontiers)} {}
};

class SamplerObject {
 private:
  aten::CSRMatrix _csc;
  std::vector<int64_t> _fanouts;
  std::deque<std::shared_ptr<GraphBatch>> _batches;
  int64_t _pool_size{1};
  int64_t _next_id{0};

  // return the unique elements in the arr
  NDArray Unique(const std::vector<NDArray> &rows) {
    NDArray arr = aten::Concat(rows);
    int64_t num_input = arr.NumElements();
    auto ctx = arr->ctx;
    auto stream = runtime::getCurrentCUDAStream();
    auto device = runtime::DeviceAPI::Get(ctx);
    int64_t *d_num_item =
        static_cast<int64_t *>(device->AllocWorkspace(ctx, sizeof(int64_t)));

    int64_t h_num_item = 0;
    NDArray unique = NDArray::Empty({num_input}, arr->dtype, ctx);
    ATEN_ID_TYPE_SWITCH(arr->dtype, IdType, {
      auto hash_table =
          runtime::cuda::OrderedHashTable<IdType>(num_input, ctx, stream);
      hash_table.FillWithDuplicates(
          arr.Ptr<IdType>(), num_input, unique.Ptr<IdType>(), d_num_item,
          stream);
      CUDA_CALL(cudaMemcpyAsync(
          &h_num_item, d_num_item, sizeof(int64_t), cudaMemcpyDeviceToHost,
          stream));
      device->StreamSync(ctx, stream);
    });

    // LOG(INFO) << "num_input: " << num_input << " num_unique: " << h_num_item;
    return unique.CreateView({h_num_item}, arr->dtype);
  }

 public:
  static std::shared_ptr<SamplerObject> const Global() {
    static auto single_instance = std::make_shared<SamplerObject>();
    return single_instance;
  }

  void SetGraph(NDArray indptr, NDArray indices, NDArray data) {
    CHECK(indptr.IsPinned() || indptr->ctx.device_id != 0)
        << "Indptr must be pinned or on gpu";
    CHECK(indices.IsPinned() || indices->ctx.device_id != 0)
        << "Indices must be pinned or on gpu";
    CHECK(
        data.IsPinned() || data->ctx == indices->ctx || data.NumElements() == 0)
        << "Data must be either empty or pinned or on gpu";
    CHECK(
        data.NumElements() == indices.NumElements() || data.NumElements() == 0);
    int64_t nrows = indptr.NumElements() - 1;
    int64_t ncols = indices.NumElements();
    if (data.NumElements() == 0)
      _csc = aten::CSRMatrix(nrows, ncols, indptr, indices);
    else
      _csc = aten::CSRMatrix(nrows, ncols, indptr, indices, data);
  }

  void SetFanout(std::vector<int64_t> fanouts) { _fanouts = fanouts; }
  void SetPoolSize(int64_t pool_size) { _pool_size = pool_size; }

  /*
   * seeds: the root vertex to be sampled from the graph
   * replace: use replace sampling or not
   */
  int64_t SampleBatch(NDArray seeds, bool replace) {
    CHECK(seeds.IsPinned() || seeds->ctx.device_type != DGLDeviceType::kDGLCPU)
        << "Seeds must be pinned or in GPU memory";
    while (_batches.size() >= _pool_size) _batches.pop_front();
    std::vector<aten::COOMatrix> blocks;
    std::vector<NDArray> frontiers = {seeds};

    for (size_t layer = 0; layer < _fanouts.size(); layer++) {
      int64_t fanout = _fanouts.at(layer);
      auto frontier = frontiers.at(layer);
      aten::COOMatrix block = aten::CSRRowWiseSampling(
          _csc, frontier, fanout, aten::NullArray(), replace);
      blocks.push_back(block);
      frontiers.push_back(Unique({frontier, block.col}));
    }
    int64_t batch_id = _next_id;
    _batches.push_back(
        std::make_shared<GraphBatch>(batch_id, frontiers, blocks));
    _next_id++;
    return batch_id;
  }

  HeteroGraphRef GetBlock(
      int64_t batch_id, int64_t layer, bool should_reindex = true) {
    // LOG(INFO) << "GetBlock batch id: "<< batch_id << " layer: " << layer;
    CHECK(layer < _fanouts.size());
    std::shared_ptr<GraphBatch> batch{nullptr};
    bool found = false;
    for (auto ptr : _batches) {
      if (ptr->_batch_id == batch_id) {
        batch = ptr;
        found = true;
        break;
      }
    }
    CHECK(found) << "GetBlock batch id " << batch_id << " is not found in the pool";
    CHECK(batch->_blocks.size() == _fanouts.size());
    if (should_reindex && !batch->reindexed) {
      auto allnodes = batch->_frontiers.at(_fanouts.size());
      ATEN_ID_TYPE_SWITCH(allnodes->dtype, IdType, {
        int64_t num_input = allnodes.NumElements();
        auto ctx = allnodes->ctx;
        auto stream = runtime::getCurrentCUDAStream();
        auto hash_table =
            runtime::cuda::OrderedHashTable<IdType>(num_input, ctx, stream);
        hash_table.FillWithUnique(allnodes.Ptr<IdType>(), num_input, stream);
        for (auto &block : batch->_blocks) {
          GPUMapEdges<IdType>(block, hash_table);
        }
      });
      batch->reindexed = true;
    }

    if (batch->_blockrefs.size() != batch->_blocks.size()) {
      // create graph ref
      for (size_t i = 0; i < _fanouts.size(); i++) {
        auto & mat = batch->_blocks.at(i);
        int64_t num_src = batch->_frontiers.at(i + 1).NumElements();
        int64_t num_dst = batch->_frontiers.at(i).NumElements();
        auto graph = HeteroGraphRef(CreateFromCOO(2, num_src, num_dst, mat.col, mat.row, false, false, COO_CODE));
        batch->_blockrefs.push_back(graph);
      }
    }
    CHECK_EQ(batch->_blockrefs.size(), batch->_blocks.size());
    return batch->_blockrefs.at(layer);
  }

  NDArray GetInputNode(int64_t batch_id) {
    std::shared_ptr<GraphBatch> batch{nullptr};
    bool found = false;
    for (auto ptr : _batches) {
      if (ptr->_batch_id == batch_id) {
        batch = ptr;
        found = true;
        break;
      }
    }
    CHECK(found) << "GetInputNode batch id " << batch_id << " is not found in the pool";
    CHECK(batch->_frontiers.size() == _fanouts.size() + 1);
    return batch->_frontiers.at(_fanouts.size());
  }
  NDArray GetOutputNode(int64_t batch_id) {
    std::shared_ptr<GraphBatch> batch{nullptr};
    bool found = false;
    for (auto ptr : _batches) {
      if (ptr->_batch_id == batch_id) {
        batch = ptr;
        found = true;
        break;
      }
    }
    CHECK(found) << "GetOutputNode batch id " << batch_id << " is not found in the pool";
    CHECK(batch->_frontiers.size() >= 1);
    return batch->_frontiers.at(0);
  }
};

}  // namespace dgl::dev
#endif  // DGL_SAMPLER_H
