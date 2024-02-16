//
// Created by juelin on 2/13/24.
//

#ifndef DGL_SPLIT_SAMPLER_H
#define DGL_SPLIT_SAMPLER_H
#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/container.h>

#include <memory>
#include <queue>
#include <utility>
#include <nccl.h>

#include "../graph/unit_graph.h"
#include "../runtime/cuda/cuda_common.h"
#include "../runtime/cuda/cuda_hashtable.cuh"
#include "array_scatter.h"
#include "cuda/bitmap.h"
#include "cuda/index_select.cuh"
#include "cuda/map_edges.cuh"
#include "cuda/all2all.h"

namespace dgl::dev {

struct SplitBatch {
  int64_t _batch_id{-1};
  int64_t _num_dp{0};

  bool reindexed{false};
  std::vector<aten::COOMatrix> _blocks;
  std::vector<NDArray> _frontiers;
  std::vector<NDArray> _upper_index;
  std::vector<ScatteredArray> _scattered_arrays;
  std::vector<HeteroGraphRef> _blockrefs;
  SplitBatch() = default;
  SplitBatch(
      int64_t batch_id, int64_t num_dp, const std::vector<NDArray>& frontiers,
      const std::vector<aten::COOMatrix>& blocks,
      const std::vector<ScatteredArray>& scattered_arrays)
      : _batch_id{batch_id},
        _num_dp{num_dp},
        _blocks{blocks},
        _frontiers{frontiers},
        _scattered_arrays{scattered_arrays} {}
};

class SplitSampler {
 private:
  aten::CSRMatrix _csc;
  NDArray _partition_map;
  std::vector<int64_t> _fanouts;
  std::shared_ptr<SplitBatch> _batch;
  int64_t _v_num{0};
  int64_t _e_num{0};
  int64_t _next_id{0};
  int64_t _rank{0};
  int64_t _world_size{0};
  int64_t _num_partitions{0};
  int64_t _num_dp{0};  // number of dp layers
  ncclComm_t _nccl_comm{nullptr};
  bool _use_bitmap{false};
  DGLContext _ctx{};  // inferred from the ctx of the seeds

  // return the unique elements in the arr
  NDArray getUnique(const std::vector<NDArray>& rows) const {
    if (_use_bitmap) {
      CHECK(!rows.empty());
      //      auto ctx = rows.at(0)->ctx;
      auto dtype = rows.at(0)->dtype;
      int64_t v_num = _csc.indptr.NumElements() - 1;
//      DeviceBitmap bitmap(v_num, _ctx);
      auto bitmap = getBitmap(v_num, _ctx);
      ATEN_ID_TYPE_SWITCH(rows.at(0)->dtype, IdType, {
        int64_t num_input{0};
        for (auto const& row : rows) {
          bitmap->flag(row.Ptr<IdType>(), row.NumElements());
          num_input += row.NumElements();
        }
        NDArray ret = NDArray::Empty({num_input}, dtype, _ctx);
        int64_t num_unique = bitmap->unique(ret.Ptr<IdType>());
        return ret.CreateView({num_unique}, dtype);
      });
    } else {
      NDArray arr = aten::Concat(rows);
      int64_t num_input = arr.NumElements();
      //      auto ctx = arr->ctx;
      auto stream = runtime::getCurrentCUDAStream();
      auto device = runtime::DeviceAPI::Get(_ctx);
      auto* d_num_item =
          static_cast<int64_t*>(device->AllocWorkspace(_ctx, sizeof(int64_t)));

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
  ~SplitSampler() {
    if (_nccl_comm) ncclCommDestroy(_nccl_comm);
  }

  static std::shared_ptr<SplitSampler> Global() {
    static auto single_instance = std::make_shared<SplitSampler>();
    return single_instance;
  }

  void initNcclComm(int64_t nranks, ncclUniqueId commId, int64_t rank){
    auto res = ncclCommInitRank(&_nccl_comm, nranks, commId, rank);
    CHECK_EQ(res, ncclSuccess);
  }

  void setGraph(
      const NDArray& indptr, const NDArray& indices, const NDArray& data) {
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
    _v_num = nrows;
    _e_num = ncols;
  }

  void setPartitionMap(const NDArray& partition_map) {
    _partition_map = partition_map;
  }

  void setFanouts(std::vector<int64_t> fanouts) {
    _fanouts = std::move(fanouts);
    _next_id = 0;
  }

  void useBitmap(bool use_bitmap) { _use_bitmap = use_bitmap; }

  void setRank(int64_t rank, int64_t world_size) {
    _rank = rank;
    _world_size = world_size;
  }

  void setNumPartitions(int64_t num_partitions) {
    _num_partitions = num_partitions;
  }

  void setNumDP(int64_t num_dp) { _num_dp = num_dp; }

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
    std::vector<ScatteredArray> scattered_arrays;
    for (size_t layer = 0; layer < _fanouts.size(); layer++) {
      int64_t fanout = _fanouts.at(layer);
      auto frontier = frontiers.at(layer);
      // assume seeds are all local nodes
      aten::COOMatrix block = aten::CSRRowWiseSampling(
          _csc, frontier, fanout, aten::NullArray(), replace);
      blocks.push_back(block);

      if (layer < _num_dp) {
        frontiers.push_back(getUnique({frontier, block.col}));
      } else {
        auto unique_src = getUnique({frontier, block.col});
        auto partition_idx = IndexSelect(_partition_map, unique_src, runtime::getCurrentCUDAStream());
        auto scatter_arr =
            ScatteredArray::Create(_v_num, _ctx, unique_src->dtype, _nccl_comm);
        // send remote frontiers to remote gpus
        // and receive frontiers from other gpus
        // build corresponding indices for mapping
        Scatter(
            _rank, _world_size, _num_partitions, unique_src, partition_idx,
            scatter_arr);
        frontiers.push_back(scatter_arr->unique_array.CreateView(
            {scatter_arr->_unique_dim}, scatter_arr->unique_array->dtype));
        scattered_arrays.push_back(scatter_arr);
      }
    }
    int64_t batch_id = _next_id++;
    _batch = std::make_shared<SplitBatch>(
        batch_id, _num_dp, frontiers, blocks, scattered_arrays);
    return batch_id;
  }

  HeteroGraphRef getBlock(
      int64_t batch_id, int64_t layer, bool should_reindex = true) {
    CHECK(layer < _fanouts.size());
    std::shared_ptr<SplitBatch> batch{_batch};
    CHECK_EQ(batch->_batch_id, batch_id)
        << "getBlock batch id " << batch_id << " is not found in the pool";
    CHECK(batch->_blocks.size() == _fanouts.size());
    if (should_reindex && !batch->reindexed) {
      auto all_nodes = batch->_frontiers.at(_fanouts.size());
      ATEN_ID_TYPE_SWITCH(all_nodes->dtype, IdType, {
        int64_t num_input = all_nodes.NumElements();
        auto stream = runtime::getCurrentCUDAStream();
        if (_use_bitmap) {
          int64_t v_num = _csc.indptr.NumElements() - 1;
//          DeviceBitmap bitmap(v_num, _ctx);
          auto bitmap = getBitmap(_v_num, _ctx);
          bitmap->flag(all_nodes.Ptr<IdType>(), all_nodes.NumElements());
          bitmap->buildOffset();
          assert(bitmap->numItem() == num_input);
          for (auto& block : batch->_blocks) {
            bitmap->map(
                block.col.Ptr<IdType>(), block.col.NumElements(),
                block.col.Ptr<IdType>());
            bitmap->map(
                block.row.Ptr<IdType>(), block.row.NumElements(),
                block.row.Ptr<IdType>());
          }
        } else {
          auto hash_table =
              runtime::cuda::OrderedHashTable<IdType>(num_input, _ctx, stream);
          hash_table.FillWithUnique(all_nodes.Ptr<IdType>(), num_input, stream);
          for (auto& block : batch->_blocks) {
            GPUMapEdges<IdType>(block, hash_table, stream);
          }
        }
      });
      batch->reindexed = true;
    }

    if (batch->_blockrefs.size() != batch->_blocks.size()) {
      // create graph ref
      for (size_t i = 0; i < _fanouts.size(); i++) {
        auto& mat = batch->_blocks.at(i);
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
    std::shared_ptr<SplitBatch> batch{_batch};
    CHECK_EQ(batch->_batch_id, batch_id);
    CHECK(batch->_frontiers.size() == _fanouts.size() + 1);
    return batch->_frontiers.at(_fanouts.size());
  }
  NDArray getOutputNode(int64_t batch_id) {
    std::shared_ptr<SplitBatch> batch{_batch};
    CHECK_EQ(batch->_batch_id, batch_id);
    CHECK(!batch->_frontiers.empty());
    return batch->_frontiers.at(0);
  }
  NDArray getBlockData(int64_t batch_id, int64_t layer) {
    std::shared_ptr<SplitBatch> batch{_batch};
    CHECK_EQ(batch->_batch_id, batch_id);
    return batch->_blocks.at(layer).data;
  }
};

}  // namespace dgl::dev
#endif  // DGL_SPLIT_SAMPLER_H
