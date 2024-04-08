//
// Created by juelinliu on 12/16/23.
//

#include <cub/cub.cuh>

#include "batch_rowwise_sampling.cuh"
#include "map_edges.cuh"

// namespace dgl::runtime::cuda {
// template class OrderedHashTable<int32_t>;
// template class OrderedHashTable<int64_t>;
// }  // namespace dgl::runtime::cuda

namespace dgl::dev {
constexpr size_t BLOCK_SIZE = 32;
constexpr size_t TILE_SIZE = 1;  // number of rows each block will cover
size_t TableSize(const size_t num, const int scale = 1) {
  const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  return next_pow2 << scale;
}

namespace impl {
/**
 * @brief Compute the size of each row in the sampled CSR, without replacement.
 *
 * @tparam IdType The type of node and edge indexes.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The index where each row's edges start.
 * @param out_deg The size of each row in the sampled matrix, as indexed by
 * `in_rows` (output).
 */
template <typename IdType>
__global__ void _CSRRowWiseDegreeKernel(
    const int64_t num_rows, const IdType *const in_rows,
    const IdType *const in_ptr, IdType *const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int in_row = in_rows[tIdx];
    const int out_row = tIdx;
    out_deg[out_row] = in_ptr[in_row + 1] - in_ptr[in_row];

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}  // _CSRRowWiseDegreeKernel

/**
 * @brief Perform row-wise uniform sampling on a CSR matrix,
 * and generate a COO matrix, with replacement.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 * @param num_rows The number of rows to pick.
 * @param num_rows The number of cols to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_cols The indices array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param out_cols The columns of the output COO (output).
 */

template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseLoadingKernel(
    const int64_t num_rows, const int64_t num_cols, const IdType *in_rows,
    const IdType *in_ptr, const IdType *in_cols, const IdType *in_data,
    const IdType *out_ptr, IdType *const out_cols, IdType *const out_data) {
  // assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);
  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);
  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_ptr[row];
    const int64_t out_row_start = out_ptr[out_row];
    assert(out_row_start < num_cols);
    for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
      const IdType in_idx = in_row_start + idx;
      const IdType out_idx = out_row_start + idx;
      out_cols[out_idx] = in_cols[in_idx];
      out_data[out_idx] = (in_data == nullptr) ? in_idx : in_data[in_idx];
    }
    out_row += 1;
  }
}  // _CSRRowWiseLoadingKernel
}  // namespace impl

template <DGLDeviceType XPU, typename IdType>
std::vector<aten::COOMatrix> CSRRowWiseSamplingUniformBatch(
    const aten::CSRMatrix &mat, const std::vector<NDArray> &rows,
    const int64_t num_picks, const bool replace) {
  auto stream = runtime::getCurrentCUDAStream();
  auto ctx = rows.at(0)->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  auto idtype = rows.at(0)->dtype;
  CHECK_EQ(idtype, mat.indices->dtype);
  CHECK_EQ(idtype, mat.indptr->dtype);
  const IdType *in_ptr;
  const IdType *in_cols;
  int64_t v_num = mat.indptr.NumElements() - 1;
  int64_t num_batches = rows.size();
  ByteMap bitmap{v_num, ctx};
  if (mat.indptr.IsPinned()) {
    void *ptr = mat.indptr->data;
    CUDA_CALL(cudaHostGetDevicePointer(&ptr, ptr, 0));
    in_ptr = static_cast<IdType *>(ptr);
  } else {
    in_ptr = static_cast<IdType *>(mat.indptr->data);
  }

  if (mat.indices.IsPinned()) {
    void *ptr = mat.indices->data;
    CUDA_CALL(cudaHostGetDevicePointer(&ptr, ptr, 0));
    in_cols = static_cast<IdType *>(ptr);
  } else {
    in_cols = static_cast<IdType *>(mat.indices->data);
  }

  int64_t num_elems = 0;
  for (const auto &row : rows) {
    bitmap.Mask(row);
    num_elems += row.NumElements();
  }

  NDArray allnodes = bitmap.Flagged(idtype);

  int64_t num_rows = allnodes.NumElements();
  if (rows.size() == 1) {
    CHECK_EQ(num_rows, rows.at(0).NumElements());
  } else {
    CHECK_LE(num_rows, num_elems);
  }

  auto hash_table =
      runtime::cuda::OrderedHashTable<IdType>(num_rows, ctx, stream);
  hash_table.FillWithUnique(
      allnodes.Ptr<IdType>(), allnodes.NumElements(), stream);

  NDArray out_deg_nd = NDArray::Empty({num_rows + 1}, idtype, ctx);
  NDArray out_indptr = NDArray::Empty({num_rows + 1}, idtype, ctx);
  IdType *out_deg = out_deg_nd.Ptr<IdType>();
  const IdType *slice_rows = allnodes.Ptr<IdType>();
  {
    const dim3 block(256);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
        impl::_CSRRowWiseDegreeKernel, grid, block, 0, stream, num_rows,
        slice_rows, in_ptr, out_deg);
  }

  IdType *out_ptr = static_cast<IdType *>(out_indptr->data);
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void *prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->StreamSync(ctx, stream);

  IdType num_cols{0};
  CUDA_CALL(cudaMemcpyAsync(
      &num_cols, out_ptr + num_rows, sizeof(IdType), cudaMemcpyDeviceToHost,
      stream));
  device->StreamSync(ctx, stream);

  NDArray out_indices = NDArray::Empty({num_cols}, idtype, ctx);
  NDArray out_databuf = NDArray::Empty({num_cols}, idtype, ctx);
  const IdType *in_data =
      (mat.data.NumElements() == 0) ? nullptr : mat.data.Ptr<IdType>();
  IdType *out_cols = out_indices.Ptr<IdType>();
  IdType *out_data = out_databuf.Ptr<IdType>();
  {
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
        (impl::_CSRRowWiseLoadingKernel<IdType, TILE_SIZE>), grid, block, 0,
        stream, num_rows, num_cols, slice_rows, in_ptr, in_cols, in_data,
        out_ptr, out_cols, out_data);
  }

  aten::CSRMatrix cached_mat =
      aten::CSRMatrix{num_rows, num_cols, out_indptr, out_indices, out_databuf};

  std::vector<aten::COOMatrix> ret;

  for (const auto &frontier : rows) {
    // map ids in the frontier to their corresponding local ids in cached mat

    NDArray loc_frontier = NDArray::Empty(
        {frontier.NumElements()}, frontier->dtype, frontier->ctx);

    GPUMapEdges(frontier, loc_frontier, hash_table, stream);
    device->StreamSync(ctx, stream);

    auto block = aten::CSRRowWiseSampling(
        cached_mat, loc_frontier, num_picks, aten::NullArray(), replace);
    // map ids in the block.row back to their corresponding global ids in mat
    block.row = aten::IndexSelect(allnodes, block.row);
    ret.push_back(block);
  }
  return ret;
};

template std::vector<aten::COOMatrix>
CSRRowWiseSamplingUniformBatch<kDGLCUDA, int32_t>(
    const aten::CSRMatrix &, const std::vector<NDArray> &, const int64_t,
    const bool);
template std::vector<aten::COOMatrix>
CSRRowWiseSamplingUniformBatch<kDGLCUDA, int64_t>(
    const aten::CSRMatrix &, const std::vector<NDArray> &, const int64_t,
    const bool);
}  // namespace dgl::dev