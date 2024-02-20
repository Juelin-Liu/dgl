//
// Created by juelinliu on 12/10/23.
//

#ifndef DGL_SAMPLER_H
#define DGL_SAMPLER_H
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
#include "cuda/bitmap.h"
#include "cuda/map_edges.cuh"
namespace dgl::dev {
struct GraphBatch {
  int64_t _batch_id{-1};
  bool reindexed{false};
  std::vector<aten::COOMatrix> _blocks;
  std::vector<NDArray> _frontiers;
  std::vector<HeteroGraphRef> _blockrefs;
  GraphBatch() = default;
  GraphBatch(
      int64_t batch_id, std::vector<NDArray> frontiers,
      std::vector<aten::COOMatrix> blocks)
      : _batch_id{batch_id},
        _blocks{std::move(blocks)},
        _frontiers{std::move(frontiers)} {}
};

class Sampler {
 private:
  aten::CSRMatrix _csc;
  std::vector<int64_t> _fanouts;
  std::shared_ptr<GraphBatch> _batch;
  int64_t _next_id{0};
  bool _use_bitmap{false};
  DGLContext _ctx{}; // inferred from the ctx of the seeds

  // return the unique elements in the arr
  NDArray getUnique(const std::vector<NDArray> &rows) const {
    if (_use_bitmap) {
      CHECK(!rows.empty());
//      auto ctx = rows.at(0)->ctx;
      auto dtype = rows.at(0)->dtype;
      int64_t v_num = _csc.indptr.NumElements() - 1;
      // LOG(INFO) << "bitmap v_num :" << v_num;
      DeviceBitmap bitmap(v_num, _ctx);
      ATEN_ID_TYPE_SWITCH(rows.at(0)->dtype, IdType, {
        int64_t num_input{0};
        for (auto const &row : rows) {
          bitmap.flag(row.Ptr<IdType>(), row.NumElements());
          num_input += row.NumElements();
        }
        NDArray ret = NDArray::Empty({num_input}, dtype, _ctx);
        int64_t num_unique = bitmap.unique(ret.Ptr<IdType>());
        return ret.CreateView({num_unique}, dtype);
      });
    } else {
      NDArray arr = aten::Concat(rows);
      int64_t num_input = arr.NumElements();
//      auto ctx = arr->ctx;
      auto stream = runtime::getCurrentCUDAStream();
      auto device = runtime::DeviceAPI::Get(_ctx);
      auto *d_num_item =
          static_cast<int64_t *>(device->AllocWorkspace(_ctx, sizeof(int64_t)));

      int64_t h_num_item = 0;
      NDArray unique = NDArray::Empty({num_input}, arr->dtype, _ctx);
      ATEN_ID_TYPE_SWITCH(arr->dtype, IdType, {
        auto hash_table =
            runtime::cuda::OrderedHashTable<IdType>(num_input, _ctx, stream);
        hash_table.FillWithDuplicates(
            arr.Ptr<IdType>(), num_input, unique.Ptr<IdType>(), d_num_item,
            stream);
        CUDA_CALL(cudaMemcpyAsync(
            &h_num_item, d_num_item, sizeof(int64_t), cudaMemcpyDeviceToHost,
            stream));
        device->StreamSync(_ctx, stream);
      });
      return unique.CreateView({h_num_item}, arr->dtype);
    }
  }

 public:
  static std::shared_ptr<Sampler> Global() {
    static auto single_instance = std::make_shared<Sampler>();
    return single_instance;
  }

  void setGraph(const NDArray& indptr, const NDArray& indices, const NDArray& data) {
    CHECK(indptr.IsPinned() || indptr->ctx.device_type != kDGLCPU)
        << "Indptr must be pinned or on gpu";
    CHECK(indices.IsPinned() || indices->ctx.device_type != kDGLCPU)
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
    _next_id = 0;
  }

  void setFanouts(std::vector<int64_t> fanouts) {
    _fanouts = std::move(fanouts);
    _next_id = 0;
  }

    void useBitmap(bool use_bitmap) {
      _use_bitmap = use_bitmap;
    }

  /*
   * seeds: the root vertex to be sampled from the graph
   * replace: use replace sampling or not
   */
  int64_t sampleOneBatch(const NDArray& seeds, bool replace) {
    CHECK(seeds.IsPinned() || seeds->ctx.device_type != DGLDeviceType::kDGLCPU)
        << "Seeds must be pinned or in GPU memory";
    _ctx = seeds->ctx;
    std::vector<aten::COOMatrix> blocks;
    std::vector<NDArray> frontiers = {seeds};

    for (size_t layer = 0; layer < _fanouts.size(); layer++) {
      int64_t fanout = _fanouts.at(layer);
      auto frontier = frontiers.at(layer);
      aten::COOMatrix block = aten::CSRRowWiseSampling(
          _csc, frontier, fanout, aten::NullArray(), replace);
      blocks.push_back(block);
      frontiers.push_back(getUnique({frontier, block.col}));
    }
    int64_t batch_id = _next_id;
    _batch = std::make_shared<GraphBatch>(batch_id, frontiers, blocks);
    _next_id++;
    return batch_id;
  }

  HeteroGraphRef getBlock(
      int64_t batch_id, int64_t layer, bool should_reindex = true) {
    CHECK(layer < _fanouts.size());
    std::shared_ptr<GraphBatch> batch{_batch};
    CHECK_EQ(batch->_batch_id, batch_id) << "getBlock batch id " << batch_id
                 << " is not found in the pool";
    CHECK(batch->_blocks.size() == _fanouts.size());
    if (should_reindex && !batch->reindexed) {
      auto all_nodes = batch->_frontiers.at(_fanouts.size());
      ATEN_ID_TYPE_SWITCH(all_nodes->dtype, IdType, {
        int64_t num_input = all_nodes.NumElements();
        auto stream = runtime::getCurrentCUDAStream();
        for (size_t cur_layer = 0; cur_layer < _fanouts.size(); cur_layer++) {

        if (_use_bitmap) {
          int64_t v_num = _csc.indptr.NumElements() - 1;
            DeviceBitmap bitmap(v_num, _ctx);
            auto& block = _batch->_blocks.at(cur_layer);
            const auto& unique_dst = _batch->_frontiers.at(cur_layer);
            const auto& unique_src = _batch->_frontiers.at(cur_layer+1);
            const int64_t num_advance = unique_src.NumElements() - unique_dst.NumElements();
            CHECK_GE(num_advance, 0);
            auto device = runtime::DeviceAPI::Get(_ctx);
            device->StreamSync(_ctx, stream);
            LOG(INFO) << "layer: " << cur_layer << " unique_dst: " << unique_dst.NumElements() << " unique_src: " << unique_src.NumElements() << " row: " << block.row.NumElements() << " col: " << block.col.NumElements();
            LOG(INFO) << "org col:" << block.col;
            LOG(INFO) << "org row: " << block.row;
            bitmap.flag(unique_dst.Ptr<IdType>(), unique_dst.NumElements());
            bitmap.map(block.row.Ptr<IdType>(), block.row.NumElements(), block.row.Ptr<IdType>());
            bitmap.map_uncheck(block.col.Ptr<IdType>(), block.col.NumElements(), block.col.Ptr<IdType>());

            bitmap.flag(unique_src.Ptr<IdType>(), unique_src.NumElements());
            bitmap.unflag(unique_dst.Ptr<IdType>(), unique_dst.NumElements());
            bitmap.set_advance(num_advance);
            bitmap.map_uncheck(block.col.Ptr<IdType>(), block.col.NumElements(), block.col.Ptr<IdType>());
            device->StreamSync(_ctx, stream);
            LOG(INFO) << "mapped col:" << block.col;
            LOG(INFO) << "mapped row: " << block.row;
        } else {
          auto hash_table =
              runtime::cuda::OrderedHashTable<IdType>(num_input, _ctx, stream);
          hash_table.FillWithUnique(all_nodes.Ptr<IdType>(), num_input, stream);
            auto& block = _batch->_blocks.at(cur_layer);
            GPUMapEdges<IdType>(block, hash_table, stream);
          }
        }
      });
      batch->reindexed = true;
    }
    if (batch->_blockrefs.size() != batch->_blocks.size()) {
      // create graph ref
      for (size_t i = 0; i < _fanouts.size(); i++) {
        auto &mat = batch->_blocks.at(i);
        int64_t num_src = batch->_frontiers.at(i + 1).NumElements();
        int64_t num_dst = batch->_frontiers.at(i).NumElements();
        auto graph = HeteroGraphRef(CreateFromCOO(
            2, num_src, num_dst, mat.col, mat.row, false, false, COO_CODE));
        batch->_blockrefs.push_back(graph);
      }
    }
    CHECK_EQ(batch->_blockrefs.size(), batch->_blocks.size());
    return batch->_blockrefs.at(layer);
  }

  NDArray getInputNode(int64_t batch_id) {
    std::shared_ptr<GraphBatch> batch{_batch};
    CHECK_EQ(batch->_batch_id, batch_id);
    CHECK(batch->_frontiers.size() == _fanouts.size() + 1);
    return batch->_frontiers.at(_fanouts.size());
  }
  NDArray getOutputNode(int64_t batch_id) {
    std::shared_ptr<GraphBatch> batch{_batch};
    CHECK_EQ(batch->_batch_id, batch_id);
    CHECK(!batch->_frontiers.empty());
    return batch->_frontiers.at(0);
  }
  NDArray getBlockData(int64_t batch_id, int64_t layer) {
    std::shared_ptr<GraphBatch> batch{_batch};
    CHECK_EQ(batch->_batch_id, batch_id);
    return batch->_blocks.at(layer).data;
  }
};

}  // namespace dgl::dev
#endif  // DGL_SAMPLER_H
