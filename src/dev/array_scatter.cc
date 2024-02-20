#include "array_scatter.h"

#include <dgl/aten/array_ops.h>
#include <dgl/runtime/container.h>

#include <nvtx3/nvtx3.hpp>
#include <utility>

#include "../runtime/cuda/cuda_common.h"
//#include "../runtime/cuda/cuda_hashtable.cuh"
#include "cuda/all2all.h"
#include "cuda/gather.h"
#include "cuda/index_select.cuh"
#include "cuda/map_edges.cuh"
#include "cuda/partition.h"
#include "cuda/bitmap.h"

namespace dgl::dev {
using namespace runtime;

#define ATEN_PIDX_TYPE_SWITCH(val, PIDType, ...)                   \
  do {                                                          \
    CHECK_EQ((val).code, kDGLInt) << "ID must be integer type";   \
    if ((val).bits == 8) {                                                          \
      typedef int8_t PIDType;                                     \
      {__VA_ARGS__} \
    } else if ((val).bits == 16) {                                     \
      typedef int16_t PIDType;                                   \
      { __VA_ARGS__ }                                           \
    } else if ((val).bits == 32) {                                     \
      typedef int32_t PIDType;                                   \
      { __VA_ARGS__ }                                           \
    } else if ((val).bits == 64) {                              \
      typedef int64_t PIDType;                                   \
      { __VA_ARGS__ }                                           \
    } else {                                                    \
      LOG(FATAL) << "Partition index can only be, int8_t, int16_t, int32 or int64";            \
    }                                                           \
  } while (0)

template <typename IndexType>
std::tuple<IdArray, IdArray, IdArray> compute_partition_continuous_indices(
    IdArray partition_map, int num_partitions, cudaStream_t stream) {
  ATEN_PIDX_TYPE_SWITCH(partition_map->dtype, PIDType, {
    return compute_partition_continuous_indices<kDGLCUDA, IndexType, PIDType>(
        partition_map, num_partitions, stream);
  });
}

IdArray gather_atomic_accumulation(
    const IdArray &accumulated_grads, const IdArray &gather_idx_in_unique,
    const IdArray &grad_shuffled_reshape, cudaStream_t stream) {
  IdArray ret;
  ATEN_FLOAT_TYPE_SWITCH(
      accumulated_grads->dtype, IdType, "accumulated_grads", {
        ATEN_ID_TYPE_SWITCH(gather_idx_in_unique->dtype, IdxType, {
          ret = gather_atomic_accumulate<kDGLCUDA, IdType, IdxType>(
              accumulated_grads, gather_idx_in_unique, grad_shuffled_reshape,
              stream);
        });
      });
  return ret;
}

NDArray ScatteredArrayObject::shuffle_forward(
    const NDArray &feat, int rank, int world_size) const {
  CHECK_EQ(feat->shape[0], unique_array->shape[0]);
  CHECK_EQ(feat->ndim, 2);
  CHECK_EQ(feat->shape[0], _unique_dim);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  NDArray toShuffle = NDArray::Empty(
      {gather_idx_in_unique_out_shuffled->shape[0], feat->shape[1]},
      feat->dtype, feat->ctx);

  IndexSelect(feat, gather_idx_in_unique_out_shuffled, toShuffle, stream);

  NDArray feat_shuffled;
  NDArray feat_offsets;
  cudaDeviceSynchronize();
  CHECK_EQ(
      global_src_offset.ToVector<int64_t>()[world_size],
      toShuffle->shape[0]);

  std::tie(feat_shuffled, feat_offsets) = dev::Alltoall(
      rank, world_size, toShuffle, global_src_offset, local_unique_src_offset, stream, _nccl_comm);

  const int num_nodes = feat_shuffled->shape[0] / feat->shape[1];

  NDArray feat_shuffled_reshape =
      feat_shuffled.CreateView({num_nodes, feat->shape[1]}, feat->dtype, 0);
  NDArray partDiscFeat = NDArray::Empty(
      {feat_shuffled_reshape->shape[0], feat->shape[1]}, feat_shuffled->dtype,
      feat->ctx);
  IndexSelect(
      feat_shuffled_reshape, scatter_idx, partDiscFeat,
      stream);

  CHECK_EQ(partDiscFeat->shape[0], _scatter_dim);
  return partDiscFeat;
}

NDArray ScatteredArrayObject::shuffle_backward(
    const NDArray &back_grad, int rank, int world_size) const {
  CHECK_EQ(back_grad->shape[0], _scatter_dim);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  // backgrad is part disccont
  NDArray part_cont = NDArray::Empty(
      {send_offset->shape[0], back_grad->shape[1]},
      back_grad->dtype, back_grad->ctx);
  //      atomic_accumulation(part_cont, idx_original_to_part_cont, back_grad);
  //      assert(idx_original_to_part_cont->shape[0]!=back_grad->shape[0]);
  IndexSelect(back_grad, gather_idx, part_cont, stream);

  NDArray grad_shuffled;
  NDArray grad_offsets;

  std::tie(grad_shuffled, grad_offsets) = dev::Alltoall(
      rank, world_size, part_cont, local_unique_src_offset,
      global_src_offset, stream, _nccl_comm);

  const int num_nodes = grad_shuffled->shape[0] / back_grad->shape[1];

  NDArray grad_shuffled_reshape = grad_shuffled.CreateView(
      {num_nodes, back_grad->shape[1]}, back_grad->dtype, 0);

  cudaStreamSynchronize(stream);

  // offsets are always long
  auto grad_offsets_v = grad_offsets.ToVector<int64_t>();
  CHECK_EQ(back_grad->dtype.bits, 32);

  NDArray accumulated_grads = aten::Full(
      (float)0.0, unique_array->shape[0] * back_grad->shape[1], back_grad->ctx);
  accumulated_grads = accumulated_grads.CreateView(
      {unique_array->shape[0], back_grad->shape[1]}, accumulated_grads->dtype,
      0);
  //    Todo can be optimized as self nodes are rarely written
  //    Before doing this optimization have a unit test in place for this
  gather_atomic_accumulation(
      accumulated_grads, gather_idx_in_unique_out_shuffled,
      grad_shuffled_reshape, stream);
  CHECK_EQ(accumulated_grads->shape[0], _unique_dim);
  CHECK_EQ(accumulated_grads->shape[1], back_grad->shape[1]);
  return accumulated_grads;
}

void Scatter(
    int64_t rank, int64_t world_size, int64_t num_partitions,
    const NDArray &local_unique_src, const NDArray &local_partition_idx,
    ScatteredArray array) {
  CHECK_EQ(array->dtype, local_unique_src->dtype);
  CHECK_GT(local_unique_src->shape[0], 0);
  CHECK_EQ(local_partition_idx.NumElements(), local_unique_src.NumElements());
  CHECK_EQ(num_partitions, world_size);
  CHECK_GT(world_size, 1) << "World size must be greater than 1";
  auto ctx = array->ctx;
  auto dtype = array->dtype;
  auto device = runtime::DeviceAPI::Get(ctx);

  cudaStream_t stream = runtime::getCurrentCUDAStream();
  array->_scatter_dim = local_unique_src->shape[0];
  array->local_unique_src = local_unique_src;
  array->local_part_idx = local_partition_idx;
  std::tie(
      array->local_unique_src_offset, array->gather_idx, array->scatter_idx) =
      compute_partition_continuous_indices<int64_t>(
          array->local_part_idx, num_partitions, stream);
  array->send_offset =
      IndexSelect(array->local_unique_src, array->gather_idx, stream);

  std::tie(array->global_src, array->global_src_offset) = dev::Alltoall(
      rank, world_size, array->send_offset, array->local_unique_src_offset,
      aten::NullArray(), stream, array->_nccl_comm);
//  LOG(INFO) << "rank " << rank << " start mapping";
  ATEN_ID_TYPE_SWITCH(array->global_src->dtype, IdType, {
    const auto& arr = array->global_src;
    auto num_input = array->global_src.NumElements();
    NDArray unique = NDArray::Empty({num_input}, arr->dtype, ctx);
    array->gather_idx_in_unique_out_shuffled = NDArray::Empty({num_input}, arr->dtype, ctx);

    static int64_t *d_num_item{nullptr};
    if (d_num_item == nullptr) d_num_item = static_cast<int64_t *>(device->AllocWorkspace(ctx, sizeof(int64_t)));

    int64_t h_num_item{0};
    auto hash_table =
        runtime::cuda::OrderedHashTable<IdType>(num_input, ctx, stream);
    hash_table.FillWithDuplicates(
        arr.Ptr<IdType>(), num_input, unique.Ptr<IdType>(), d_num_item,
        stream);
    CUDA_CALL(cudaMemcpyAsync(
        &h_num_item, d_num_item, sizeof(int64_t), cudaMemcpyDeviceToHost,
        stream));
    GPUMapEdges(arr, array->gather_idx_in_unique_out_shuffled, hash_table, stream);
//    device->StreamSync(ctx, stream); // not necessary since GPUMapEdges will sync the stream
    array->_unique_dim = h_num_item;
    array->unique_array = unique.CreateView({array->_unique_dim}, dtype);

//    auto bitmap = getBitmap(array->_v_num, ctx);
//    bitmap->flag(
//        array->global_src.Ptr<IdType>(), array->global_src.NumElements());
//    NDArray unique_arr =
//        NDArray::Empty({array->global_src.NumElements()}, dtype, ctx);
//    array->gather_idx_in_unique_out_shuffled =
//        NDArray::Empty({array->global_src.NumElements()}, dtype, ctx);
//
//    array->_unique_dim = bitmap->unique(unique_arr.Ptr<IdType>());
//    bitmap->map(array->global_src.Ptr<IdType>(),array->global_src.NumElements(),
//               array->gather_idx_in_unique_out_shuffled.Ptr<IdType>());
//    array->unique_array = unique_arr.CreateView({array->_unique_dim}, dtype);
  });
//  LOG(INFO) << "rank " << rank << " done mapping";
}
}  // namespace dgl::dev
