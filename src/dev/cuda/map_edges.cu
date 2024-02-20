//
// Created by juelinliu on 12/11/23.
//
#include "map_edges.cuh"
#include <cassert>

using namespace dgl::runtime::cuda;
namespace dgl::runtime::cuda {
template class OrderedHashTable<int32_t>;
template class OrderedHashTable<int64_t>;
}  // namespace dgl::runtime::cuda

namespace dgl::dev {
namespace impl {

template <typename IdType>
inline size_t RoundUpDiv(const IdType num, const size_t divisor) {
  return static_cast<IdType>(num / divisor) + (num % divisor == 0 ? 0 : 1);
}

template <typename IdType>
inline IdType RoundUp(const IdType num, const size_t unit) {
  return RoundUpDiv(num, unit) * unit;
}

template <typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__device__ void map_vertex_ids(
    const IdType* const global, IdType* const new_global,
    const IdType num_vertices, const DeviceOrderedHashTable<IdType>& table) {
  assert(BLOCK_SIZE == blockDim.x);
  using Mapping = typename OrderedHashTable<IdType>::Mapping;

  const IdType tile_start = TILE_SIZE * blockIdx.x;
  const IdType tile_end = min(TILE_SIZE * (blockIdx.x + 1), num_vertices);

  for (IdType idx = threadIdx.x + tile_start; idx < tile_end;
       idx += BLOCK_SIZE) {
    const Mapping& mapping = *table.Search(global[idx]);
    new_global[idx] = mapping.local;
  }
}

/**
 * @brief Generate mapped edge endpoint ids.
 *
 * @tparam IdType The type of id.
 * @tparam BLOCK_SIZE The size of each thread block.
 * @tparam TILE_SIZE The number of edges to process per thread block.
 * @param global_srcs_device The source ids to map.
 * @param new_global_srcs_device The mapped source ids (output).
 * @param global_dsts_device The destination ids to map.
 * @param new_global_dsts_device The mapped destination ids (output).
 * @param num_edges The number of edges to map.
 * @param src_mapping The mapping of sources ids.
 * @param dst_mapping The mapping of destination ids.
 */
template <typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__global__ void map_edge_ids(
    const IdType* const global_srcs_device,
    IdType* const new_global_srcs_device,
    const IdType* const global_dsts_device,
    IdType* const new_global_dsts_device, const IdType num_edges,
    DeviceOrderedHashTable<IdType> src_mapping,
    DeviceOrderedHashTable<IdType> dst_mapping) {
  assert(BLOCK_SIZE == blockDim.x);
  assert(2 == gridDim.y);

  if (blockIdx.y == 0) {
    map_vertex_ids<IdType, BLOCK_SIZE, TILE_SIZE>(
        global_srcs_device, new_global_srcs_device, num_edges, src_mapping);
  } else {
    map_vertex_ids<IdType, BLOCK_SIZE, TILE_SIZE>(
        global_dsts_device, new_global_dsts_device, num_edges, dst_mapping);
  }
}

/**
 * @brief Generate mapped edge endpoint ids.
 *
 * @tparam IdType The type of id.
 * @tparam BLOCK_SIZE The size of each thread block.
 * @tparam TILE_SIZE The number of edges to process per thread block.
 * @param global_srcs_device The source ids to map.
 * @param new_global_srcs_device The mapped source ids (output).
 * @param global_dsts_device The destination ids to map.
 * @param new_global_dsts_device The mapped destination ids (output).
 * @param num_edges The number of edges to map.
 * @param src_mapping The mapping of sources ids.
 * @param dst_mapping The mapping of destination ids.
 */
template <typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__global__ void map_edge_ids(
    const IdType* const global_srcs_device,
    IdType* const new_global_srcs_device, const IdType num_edges,
    DeviceOrderedHashTable<IdType> src_mapping) {
  assert(BLOCK_SIZE == blockDim.x);
  assert(1 == gridDim.y);
  map_vertex_ids<IdType, BLOCK_SIZE, TILE_SIZE>(
      global_srcs_device, new_global_srcs_device, num_edges, src_mapping);
}

}  // namespace impl
template <typename IdType>
void GPUMapEdges(
    aten::COOMatrix& mat, const OrderedHashTable<IdType>& hash,
    cudaStream_t stream) {
  constexpr const int BLOCK_SIZE = 128;
  constexpr const size_t TILE_SIZE = 1024;
  int64_t num_edges = mat.col.NumElements();
  const dim3 grid(impl::RoundUpDiv<IdType>(num_edges, TILE_SIZE), 2);
  const dim3 block(BLOCK_SIZE);
  // map the srcs
  CUDA_KERNEL_CALL(
      (impl::map_edge_ids<IdType, BLOCK_SIZE, TILE_SIZE>), grid, block, 0,
      stream, mat.row.Ptr<IdType>(), mat.row.Ptr<IdType>(),
      mat.col.Ptr<IdType>(), mat.col.Ptr<IdType>(), num_edges,
      hash.DeviceHandle(), hash.DeviceHandle());
};

template <typename IdType>
void GPUMapEdges(
    const NDArray& in_rows, NDArray& ret_row,
    const runtime::cuda::OrderedHashTable<IdType>& hash_table,
    cudaStream_t stream) {
  CHECK_GE(ret_row.NumElements(), in_rows.NumElements());
  constexpr const int BLOCK_SIZE = 128;
  constexpr const size_t TILE_SIZE = 1024;
  int64_t nrows = in_rows.NumElements();
  const size_t num_tiles = (nrows + TILE_SIZE - 1) / TILE_SIZE;
  const dim3 grid(num_tiles, 1);
  const dim3 block(BLOCK_SIZE);
  auto table = hash_table.DeviceHandle();
  // map the srcs
  impl::map_edge_ids<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
      in_rows.Ptr<IdType>(), ret_row.Ptr<IdType>(), nrows, table);
  auto device = runtime::DeviceAPI::Get(in_rows->ctx);
  device->StreamSync(in_rows->ctx, stream);
};
template void GPUMapEdges<int32_t>(
    aten::COOMatrix&, const OrderedHashTable<int32_t>&, cudaStream_t);
template void GPUMapEdges<int64_t>(
    aten::COOMatrix&, const OrderedHashTable<int64_t>&, cudaStream_t);

template void GPUMapEdges<int32_t>(
    const NDArray& , NDArray& ,
    const OrderedHashTable<int32_t>& ,
    cudaStream_t );

template void GPUMapEdges<int64_t>(
    const NDArray& , NDArray& ,
    const OrderedHashTable<int64_t>& ,
    cudaStream_t );

NDArray getUnique(const std::vector<NDArray> &rows) {
  NDArray arr = aten::Concat(rows);
  int64_t num_input = arr.NumElements();
  auto ctx = arr->ctx;
  auto stream = runtime::getCurrentCUDAStream();
  auto device = runtime::DeviceAPI::Get(ctx);
  static int64_t *d_num_item{nullptr};
  if (d_num_item == nullptr) d_num_item = static_cast<int64_t *>(device->AllocWorkspace(ctx, sizeof(int64_t)));

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
  return unique.CreateView({h_num_item}, arr->dtype);
}
}  // namespace dgl::dev
