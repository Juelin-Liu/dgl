#include "array_scatter.h"

#include <dgl/aten/array_ops.h>
#include <dgl/runtime/container.h>

#include <nvtx3/nvtx3.hpp>
#include <utility>

#include "../runtime/cuda/cuda_common.h"
#include "../runtime/cuda/cuda_hashtable.cuh"
#include "cuda/all2all.h"
#include "cuda/index_select.cuh"
#include "cuda/map_edges.cuh"
#include "cuda/gather.h"
#include "cuda/partition.h"

namespace dgl::dev {
using namespace runtime;

// Index type should be 64 bits for all 2 all
typedef int64_t IndexType;

template <typename IndexType>
std::tuple<IdArray, IdArray, IdArray> compute_partition_continuous_indices(
    IdArray partition_map, int num_partitions, cudaStream_t stream) {
  std::tuple<IdArray, IdArray, IdArray> ret;
  ATEN_ID_TYPE_SWITCH(partition_map->dtype, IdType, {
    ret =
        compute_partition_continuous_indices<kDGLCUDA, IndexType, IdType>(
            partition_map, num_partitions, stream);
  });
  return ret;
}

std::tuple<IdArray, IdArray, IdArray>
compute_partition_continuous_indices_strawman(
    const IdArray &partition_map, int num_partitions, cudaStream_t stream) {
  std::tuple<IdArray, IdArray, IdArray> ret;
  ATEN_ID_TYPE_SWITCH(partition_map->dtype, IdType, {
    ret = compute_partition_continuous_indices_strawman<kDGLCUDA, IdType>(
        partition_map, num_partitions, stream);
  });
  return ret;
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
    const NDArray& feat, int rank, int world_size) const {
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
      shuffled_recv_offsets.ToVector<int64_t>()[world_size],
      toShuffle->shape[0]);

  std::tie(feat_shuffled, feat_offsets) = dev::Alltoall(
      rank, world_size, toShuffle, feat->shape[1], shuffled_recv_offsets,
      partitionContinuousOffsets, stream);

  const int num_nodes = feat_shuffled->shape[0] / feat->shape[1];

  NDArray feat_shuffled_reshape =
      feat_shuffled.CreateView({num_nodes, feat->shape[1]}, feat->dtype, 0);
  NDArray partDiscFeat = NDArray::Empty(
      {feat_shuffled_reshape->shape[0], feat->shape[1]}, feat_shuffled->dtype,
      feat->ctx);
  IndexSelect(
      feat_shuffled_reshape, scatter_idx_in_part_disc_cont, partDiscFeat,
      stream);

  CHECK_EQ(partDiscFeat->shape[0], _scatter_dim);
  return partDiscFeat;
}

NDArray ScatteredArrayObject::shuffle_backward(
    const NDArray& back_grad, int rank, int world_size) const {
  CHECK_EQ(back_grad->shape[0], _scatter_dim);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  // backgrad is part disccont
  NDArray part_cont = NDArray::Empty(
      {partitionContinuousArray->shape[0], back_grad->shape[1]},
      back_grad->dtype, back_grad->ctx);
  //      atomic_accumulation(part_cont, idx_original_to_part_cont, back_grad);
  //      assert(idx_original_to_part_cont->shape[0]!=back_grad->shape[0]);
  IndexSelect(back_grad, gather_idx_in_part_disc_cont, part_cont, stream);

  NDArray grad_shuffled;
  NDArray grad_offsets;

  std::tie(grad_shuffled, grad_offsets) = dev::Alltoall(
      rank, world_size, part_cont, back_grad->shape[1],
      partitionContinuousOffsets, shuffled_recv_offsets, stream);

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
    const NDArray &frontier, const NDArray &_partition_map,
    ScatteredArray array) {
  CHECK_EQ(array->dtype, frontier->dtype);
  CHECK_GT(frontier->shape[0], 0);
  CHECK_LE(frontier->shape[0], array->_expect_size);
  CHECK_EQ(num_partitions, world_size);

  array->_scatter_dim = frontier->shape[0];
  array->originalArray = frontier;
  array->partitionMap = _partition_map;

  // Compute partition continuous array
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  // Todo: Why not runtime stream
  nvtxRangePushA("upper_index");
  std::tuple<IdArray, IdArray, IdArray> out;

  out = compute_partition_continuous_indices<IndexType>(
      array->partitionMap, num_partitions, stream);

  const auto
      &[boundary_offsets, gather_idx_in_part_disc_cont,
        scatter_idx_in_part_disc_cont] = out;

  nvtxRangePop();
  nvtxRangePushA("create_message");
  array->partitionContinuousArray =
      IndexSelect(frontier, gather_idx_in_part_disc_cont, stream);
  array->gather_idx_in_part_disc_cont = gather_idx_in_part_disc_cont;
  array->scatter_idx_in_part_disc_cont = scatter_idx_in_part_disc_cont;
  array->partitionContinuousOffsets = boundary_offsets;
  nvtxRangePop();

  if (array->debug) {
    cudaStreamSynchronize(stream);
    CHECK_EQ(
        boundary_offsets.ToVector<int64_t>()[4],
        array->partitionContinuousArray->shape[0]);
  }
  nvtxRangePushA("shuffle");
  if (world_size != 1) {
    std::tie(array->shuffled_array, array->shuffled_recv_offsets) =
        dev::Alltoall(
            rank, world_size, array->partitionContinuousArray, 1,
            boundary_offsets, aten::NullArray(), stream);
  } else {
    array->shuffled_array = array->partitionContinuousArray;
    array->shuffled_recv_offsets = boundary_offsets;
  }
  nvtxRangePop();
  nvtxRangePushA("lower_index");
  auto device = runtime::DeviceAPI::Get(frontier->ctx);
  auto ctx = frontier->ctx;
  auto dtype = frontier->dtype;

  ATEN_ID_TYPE_SWITCH(frontier->dtype, IdType, {
    auto *d_num_item =
        static_cast<int64_t *>(device->AllocWorkspace(ctx, sizeof(int64_t)));

    auto &shuffled_arr = array->shuffled_array;
    auto &unique_arr = array->unique_array;
    auto num_input = shuffled_arr.NumElements();
    auto table =
        runtime::cuda::OrderedHashTable<IdType>(num_input, ctx, stream);
    if (array->unique_array.NumElements() < num_input) {
      array->unique_array = NDArray::Empty({num_input}, frontier->dtype, ctx);
    }

    int64_t h_num_item = 0;

    table.FillWithDuplicates(
        shuffled_arr.Ptr<IdType>(), num_input, unique_arr.Ptr<IdType>(),
        d_num_item, stream);
    CUDA_CALL(cudaMemcpyAsync(
        &h_num_item, d_num_item, sizeof(int64_t), cudaMemcpyDeviceToHost,
        stream));
    array->gather_idx_in_unique_out_shuffled =
        IdArray::Empty({num_input}, dtype, ctx);
    GPUMapEdges(
        array->shuffled_array, array->gather_idx_in_unique_out_shuffled, table,
        stream);
    device->StreamSync(ctx, stream);
    array->_unique_dim = h_num_item;
    device->FreeWorkspace(ctx, d_num_item);
  });
  nvtxRangePop();
}
}  // namespace dgl::dev
