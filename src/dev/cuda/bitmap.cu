//
// Created by juelin on 2/6/24.
//
#include <dgl/runtime/tensordispatch.h>

#include <cub/cub.cuh>

#include "../../runtime/cuda/cuda_common.h"
#include "bitmap.h"

#define BucketWidth 32
#define BucketHighestBit 0x80000000  // 1000,0000,0000,0000,...,0000 in binary
#define BlockSize 256
#define WarpSize 32

using TensorDispatcher = dgl::runtime::TensorDispatcher;

namespace dgl::dev {

class DeviceBitIterator
    : public std::iterator<std::random_access_iterator_tag, bool> {
 private:
  BucketType *_bitmap{nullptr};
  uint32_t _num_buckets{0};
  uint32_t _offset{0};

 public:
  using self_type = DeviceBitIterator;
  using value_type = bool;
  __host__ uint32_t numBuckets() const { return _num_buckets; };
  __host__ __device__ DeviceBitIterator(
      BucketType *bitmap, uint32_t num_buckets, uint32_t offset = 0)
      : _bitmap{bitmap},
        _num_buckets{num_buckets},
        _offset{offset} {
            // printf("Constructor offset = %ld\n", _offset);
        };

  __host__ __device__ __forceinline__ self_type operator++(int) {
    self_type retval = *this;
    _offset++;
    // printf("operator++(int) offset = %ld\n", _offset);

    return retval;
  }

  __host__ __device__ __forceinline__ self_type operator++() {
    _offset++;
    // printf("operator++() offset = %ld\n", _offset);

    return *this;
  }

  __host__ __device__ __forceinline__ bool operator*() const {
    const int64_t bucket_idx = _offset / BucketWidth;
    const BucketType shift = _offset % BucketWidth;
    const BucketType mask = BucketHighestBit >> shift;
    const BucketType flag = _bitmap[bucket_idx];
    bool retval = (flag & mask);
    // printf("operator*() flag %d shift %d mask %d offset %ld ret %d\n", flag,
    // shift, mask, _offset, retval);
    return retval;
  }

  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+(Distance n) const {
    self_type retval(_bitmap, _num_buckets, _offset + n);
    // printf("operator+(Distance n) offset = %ld n=%d \n", _offset, n);

    return retval;
  }

  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+=(Distance n) const {
    _offset += n;
    // printf("operator+=(Distance n) offset = %ld n=%d \n", _offset, n);

    return *this;
  }

  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator-(Distance n) const {
    self_type retval(_bitmap, _num_buckets, _offset - n);
    // printf("operator-(Distance n) offset = %ld n=%d \n", _offset, n);

    return retval;
  }

  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator-=(Distance n) const {
    _offset -= n;
    // printf("operator-=(Distance n) offset = %ld n=%d \n", _offset, n);

    return *this;
  }

  template <typename Distance>
  __host__ __device__ __forceinline__ value_type operator[](Distance n) const {
    const int64_t bucket_idx = (_offset + n) / BucketWidth;
    const BucketType shift = (_offset + n) % BucketWidth;
    const BucketType mask = BucketHighestBit >> shift;
    const BucketType flag = _bitmap[bucket_idx];
    bool retval = (flag & mask);
    // printf("operator[] n %d flag %d shift %d mask %d offset %ld ret %d\n", n,
    // flag, shift, mask, _offset, retval);
    return retval;
  }

  template <typename Distance>
  __device__ __forceinline__ void flag(Distance n) {
    assert(n < _num_buckets * BucketWidth);
    const int64_t bucket_idx = (_offset + n) / BucketWidth;
    const BucketType shift = (_offset + n) % BucketWidth;
    const BucketType mask = BucketHighestBit >> shift;
    atomicOr(_bitmap + bucket_idx, mask);
    // printf("flag n %ld shift %d mask %d offset %ld bitmap %d\n", n, shift,
    // mask, _offset, _bitmap[bucket_idx]);
  }

  template <typename Distance>
  __device__ __forceinline__ BucketType popcnt(Distance bucket_idx) const {
    return __popc(_bitmap[bucket_idx]);
  }

  template <typename Distance>
  __device__ __forceinline__ BucketType
  popcnt(Distance bucket_idx, Distance num_bits) const {
    assert(num_bits <= BucketWidth);
    const int shift = BucketWidth - num_bits;
    const BucketType bitmap = _bitmap[bucket_idx];
    const BucketType mask = (bitmap >> shift) << shift;
    const int retval = __popc(mask);
    // printf("popcnt bucket_idx %ld num_bits %ld retval %d bitmap %d shift %d
    // mask %d\n", bucket_idx, num_bits, retval, bitmap, shift, mask);
    return retval * (num_bits > 0);
  }
  __host__ __device__ __forceinline__ bool operator==(
      const self_type &rhs) const {
    return _bitmap == rhs._bitmap && _offset == rhs._offset &&
           _num_buckets == rhs._num_buckets;
  }

  __host__ __device__ __forceinline__ bool operator!=(
      const self_type &rhs) const {
    return _bitmap != rhs._bitmap || _offset != rhs._offset ||
           _num_buckets == rhs._num_buckets;
  }
};

namespace impl {
template <typename IdType>
__global__ void flag_kernel(
    DeviceBitIterator iter, const IdType *row, int64_t num_rows) {
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tIdx < num_rows) {
    iter.flag(row[tIdx]);
  }
}

//__global__ void popcnt_kernel(
//    DeviceBitIterator iter, int64_t num_buckets, OffsetType *offset) {
//  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
//  if (tIdx < num_buckets) {
//    offset[tIdx] = iter.popcnt(tIdx);
//  }
//}

// This kernel is used to initialize the offset array.
// The n-th element in the offset array stores the number of 1-bit
// in the bitmap ranging from the CompRatio * n to CompRatio * (n + 1)-th
// bucket. For instance, if CompRatio=4, The offset[0] will store the sum of
// number of 1-bit from bitmap[0], bitmap[1], ... bitmap[3].
template <int CompRatio>
__global__ void offset_kernel(
    DeviceBitIterator iter, int64_t num_buckets, OffsetType *offset) {
  assert(blockDim.x == BlockSize && CompRatio <= 32);
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  typedef cub::WarpReduce<uint32_t, CompRatio> WarpReduce;
  __shared__
      typename WarpReduce::TempStorage temp_storage[BlockSize / CompRatio];
  uint32_t cnt{0};
  if (tIdx < num_buckets) {
    cnt = iter.popcnt(tIdx);
  }
  const int group_id = threadIdx.x / CompRatio;
  uint32_t aggregate = WarpReduce(temp_storage[group_id]).Sum(cnt);
  if (threadIdx.x % CompRatio == 0) {
    offset[tIdx / CompRatio] = aggregate;
  }
}

// This kernel is used to initialize the number of 1-bit in the bitmap.
__global__ void cnt_kernel(
    DeviceBitIterator iter, int64_t num_buckets, uint32_t *d_num_item_out) {
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  typedef cub::BlockReduce<uint32_t, BlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  uint32_t thread_data{0};
  if (tIdx < num_buckets) {
    thread_data = iter.popcnt(tIdx);
  }
  uint32_t aggregate = BlockReduce(temp_storage).Sum(thread_data);
  if (threadIdx.x == 0) {
    atomicAdd(d_num_item_out, aggregate);
  }
}

// template <typename IdType>
//__global__ void map_kernel(
//     DeviceBitIterator iter, const OffsetType *offset, const IdType *row,
//     int64_t num_rows, IdType *out_row) {
//   const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (tIdx < num_rows) {
//     const IdType id = row[tIdx];
//     assert(iter[id] == true);
//     const int64_t bucket_idx = id / BucketWidth;
//     const int64_t num_bits = id % BucketWidth + 1;
//     OffsetType start = offset[bucket_idx];
//     const IdType loc_id = start + iter.popcnt(bucket_idx, num_bits);
//     out_row[tIdx] = loc_id;
//     // printf("tIdx: %lld start %d loc_id: %d num_bits: %d\n", tIdx, start,
//     // loc_id, num_bits);
//   }
// }

// assume each row id is handled by number of threads = CompRatio
// This kernel is used to map each element in row to its corresponding local
// index in the bitmap. For instance, suppose row[0]=10, and the iter[10] is the
// second 1 bit in the bitmap, this function will map 10 to 1 and write that in
// the out_row[0].

template <typename IdType, int CompRatio>
__global__ void map_kernel(
    DeviceBitIterator iter, const OffsetType *offset, const IdType *row,
    int64_t num_rows, IdType *out_row) {
  assert(blockDim.x == BlockSize && CompRatio <= 32);
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int32_t rIdx = tIdx / CompRatio;
  const int32_t tOffset = tIdx % CompRatio;
  typedef cub::WarpReduce<int, CompRatio> WarpReduce;
  __shared__
      typename WarpReduce::TempStorage temp_storage[BlockSize / CompRatio];

  if (rIdx < num_rows) {
    const IdType id = row[rIdx];
    assert(iter[id] == true);
    const int32_t bucket_idx = id / BucketWidth;
    const int32_t total_num_bits = id % (BucketWidth * CompRatio) + 1;

    int32_t num_bits = std::max(
        0, std::min(BucketWidth, total_num_bits - tOffset * BucketWidth));
    ;
    //    if (total_num_bits - tOffset * BucketWidth >= BucketWidth) {
    //      num_bits = BucketWidth;
    //    } else if (total_num_bits - tOffset * BucketWidth >= 0) {
    //      num_bits = total_num_bits - tOffset * BucketWidth;
    //    } else {
    //      num_bits = 0;
    //    }

    const int bitcnt = iter.popcnt(bucket_idx + tOffset, num_bits);
    const int group_id = threadIdx.x / CompRatio;
    const int agg_bitcnt = WarpReduce(temp_storage[group_id]).Sum(bitcnt);
    if (tOffset == 0) {
      const int32_t offset_idx = id / (BucketWidth * CompRatio);
      out_row[rIdx] = offset[offset_idx] + agg_bitcnt;
    }
    // printf("tIdx: %lld start %d loc_id: %d num_bits: %d\n", tIdx, start,
    // loc_id, num_bits);
  }
}

__global__ void read_kernel(DeviceBitIterator iter, int64_t num_items) {
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tIdx < num_items) {
    const int flag = iter[tIdx];
    printf("tIdx: %lld mask %d\n", tIdx, flag);
  }
}
// assume prefix sum has been computed of the offset
// and the offset has length equal to num_buckets + 1
// output: the indices in bitmap marked as 1
//template <typename IdType>
//__global__ void unique_kernel(
//    const BucketType *bitmap, int64_t num_buckets, OffsetType *offset,
//    IdType *out_row) {
//  assert(offset != nullptr);
//  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
//  if (tIdx < num_buckets) {
//    const OffsetType start = offset[tIdx];
//    BucketType bits = bitmap[tIdx];
//    int cnt = 0;
//    while (bits != 0) {
//      const int lz = __clz(bits);
//      out_row[start + cnt] = BucketWidth * tIdx + lz;
//      cnt++;
//      bits ^= BucketHighestBit >> lz;
//    }
//  }
//}

// assume prefix sum has been computed of the offset
// and the offset has length equal to num_offsets
// output: the indices in bitmap marked as 1
// assume total # threads >= num_buckets
template <typename IdType, int CompRatio>
__global__ void unique_kernel(
    const BucketType *bitmap, int64_t num_buckets, const OffsetType *offset,
    IdType *out_row) {
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int64_t offset_idx = tIdx / CompRatio;
  const auto offset_start = offset[offset_idx];
  const auto offset_end = offset[offset_idx + 1];
  typedef cub::WarpScan<int, CompRatio> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[BlockSize / CompRatio];

  // short circuit reading all zero buckets to reduce memory loads
  // wrap divergence might happen of CompRatio != 32
  if (tIdx < num_buckets && offset_end > offset_start) {
    auto bits = bitmap[tIdx];
    auto cnt = __popc(bits);
    auto warp_id = threadIdx.x / CompRatio;
    WarpScan (temp_storage[warp_id]).ExclusiveSum(cnt, cnt);
    auto out_idx = offset_start + cnt;

    while (bits != 0) {
      const int lz = __clz(bits);
      out_row[out_idx++] = BucketWidth * tIdx + lz;

//      if (tIdx < 32) {
//        printf("tIdx: %ld, bits: %d, cnt: %d, warp_id:%d, out_idx: %d, out_id %ld, start %d, end %d\n", tIdx, bits, cnt, warp_id, out_idx-1, BucketWidth * tIdx + lz, offset_start, offset_end);
//      }

      bits ^= BucketHighestBit >> lz;
    }
  }
}

}  // namespace impl

DeviceBitmap::DeviceBitmap(int64_t num_elems, DGLContext ctx, int comp_ratio) {
  _comp_ratio = comp_ratio;
  CHECK(_comp_ratio % 4 == 0 || _comp_ratio == 1);
  _num_buckets = (num_elems + BucketWidth - 1) / BucketWidth;  // 32 bits per buckets
  _num_buckets = _num_buckets + _comp_ratio - _num_buckets % _comp_ratio; // make it multiple on _comp_ratio
  _num_offset = (_num_buckets + _comp_ratio - 1) / _comp_ratio + 1;
  _ctx = ctx;
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  _bitmap = static_cast<BucketType *>(
      device->AllocWorkspace(ctx, _num_buckets * sizeof(BucketType)));
  CUDA_CALL(
      cudaMemsetAsync(_bitmap, 0, _num_buckets * sizeof(BucketType), stream));
  _offset = static_cast<OffsetType *>(
      device->AllocWorkspace(ctx, _num_offset * sizeof(OffsetType)));
}

DeviceBitmap::~DeviceBitmap() {
  auto device = runtime::DeviceAPI::Get(_ctx);
  if (_bitmap) device->FreeWorkspace(_ctx, _bitmap);
  if (_offset) device->FreeWorkspace(_ctx, _offset);
}

void DeviceBitmap::reset() {
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  cudaMemsetAsync(_bitmap, 0, _num_buckets * sizeof(BucketType), stream);
  device->StreamSync(_ctx, stream);
}

template <typename IdType>
void DeviceBitmap::flag(const IdType *row, int64_t num_rows) {
  auto stream = runtime::getCurrentCUDAStream();
  auto device = runtime::DeviceAPI::Get(_ctx);

  const dim3 block(BlockSize);
  const dim3 grid((num_rows + block.x - 1) / block.x);
  DeviceBitIterator iter(_bitmap, _num_buckets, 0);
  CUDA_KERNEL_CALL(
      impl::flag_kernel, grid, block, 0, stream, iter, row, num_rows);
  _offset_built = false;
}


int64_t DeviceBitmap::buildOffset() {
  if (_offset_built) return _num_unique;
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  DeviceBitIterator iter(_bitmap, _num_buckets, 0);

  const dim3 block(BlockSize);
  const dim3 grid((_num_buckets + block.x - 1) / block.x);
  switch (_comp_ratio) {
    case 1:
      CUDA_KERNEL_CALL(
          impl::offset_kernel<1>, grid, block, 0, stream, iter, _num_buckets,
          _offset);
      break;
    case 4:
      CUDA_KERNEL_CALL(
          impl::offset_kernel<4>, grid, block, 0, stream, iter, _num_buckets,
          _offset);
      break;
    case 8:
      CUDA_KERNEL_CALL(
          impl::offset_kernel<8>, grid, block, 0, stream, iter, _num_buckets,
          _offset);
      break;
    case 16:
      CUDA_KERNEL_CALL(
          impl::offset_kernel<16>, grid, block, 0, stream, iter, _num_buckets,
          _offset);
      break;
    case 32:
      CUDA_KERNEL_CALL(
          impl::offset_kernel<32>, grid, block, 0, stream, iter, _num_buckets,
          _offset);
      break;
    default:
      LOG(ERROR) << "unsupported compression ratio, must be 1, 4, 8, 16, 32";
      break;
  }
  // use prefix sum to compute the indices
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  auto d_in = _offset;
  auto d_out = _offset;
  auto num_items = _num_offset;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream));
  //  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  d_temp_storage = device->AllocWorkspace(_ctx, temp_storage_bytes);

  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
        {1}, DGLDataTypeTraits<OffsetType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
        {1}, DGLDataTypeTraits<OffsetType>::dtype, DGLContext{kDGLCPU, 0});
  }
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream));
  CUDA_CALL(cudaMemcpyAsync(
      new_len_tensor.Ptr<OffsetType>(), _offset + _num_offset - 1,
      sizeof(OffsetType), cudaMemcpyDefault, stream));

  device->StreamSync(_ctx, stream);
  device->FreeWorkspace(_ctx, d_temp_storage);
  _offset_built = true;
  _num_unique = new_len_tensor.Ptr<OffsetType>()[0];
  return _num_unique;
}

int64_t DeviceBitmap::numItem() const {
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  const dim3 block(BlockSize);
  const dim3 grid((_num_buckets + block.x - 1) / block.x);
  DeviceBitIterator iter(_bitmap, _num_buckets, 0);
  auto *d_num_item =
      static_cast<uint32_t *>(device->AllocWorkspace(_ctx, sizeof(uint32_t)));
  uint32_t h_num_item{0};
  CUDA_CALL(cudaMemsetAsync(d_num_item, 0, sizeof(uint32_t), stream));
  CUDA_KERNEL_CALL(
      impl::cnt_kernel, grid, block, 0, stream, iter, _num_buckets, d_num_item);
  CUDA_CALL(cudaMemcpyAsync(
      &h_num_item, d_num_item, sizeof(uint32_t), cudaMemcpyDeviceToHost,
      stream));
  device->StreamSync(_ctx, stream);
  return h_num_item;
}

// template <typename IdType>
// int64_t DeviceBitmap::unique(IdType *out_row) const {
//   auto device = runtime::DeviceAPI::Get(_ctx);
//   auto stream = runtime::getCurrentCUDAStream();
//
//   int num_items = _num_buckets * sizeof(Bucket) * 8;
//   auto d_in = cub::CountingInputIterator<IdType>(0);
//   DeviceBitIterator d_flags(_bitmap, _num_buckets, 0);
//   IdType *d_out = out_row;
//   int64_t *d_num_selected_out =
//       static_cast<int64_t *>(device->AllocWorkspace(_ctx, sizeof(int64_t)));
//   void *d_temp_storage = NULL;
//   size_t temp_storage_bytes = 0;
//   CUDA_CALL(cub::DeviceSelect::Flagged(
//       d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out,
//       d_num_selected_out, num_items, stream));
//   device->StreamSync(_ctx, stream);
//   d_temp_storage = device->AllocWorkspace(_ctx, temp_storage_bytes);
//   // Run selection
//   CUDA_CALL(cub::DeviceSelect::Flagged(
//       d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out,
//       d_num_selected_out, num_items, stream));
//   int64_t h_num_selected_out{0};
//   cudaDeviceSynchronize();
//   CUDA_CALL(cudaMemcpyAsync(
//       &h_num_selected_out, d_num_selected_out, sizeof(int64_t),
//       cudaMemcpyDefault, stream));
//   device->StreamSync(_ctx, stream);
//   device->FreeWorkspace(_ctx, d_temp_storage);
//   device->FreeWorkspace(_ctx, d_num_selected_out);
//   return h_num_selected_out;
// };

template <typename IdType>
int64_t DeviceBitmap::unique(IdType *out_row) {
  int64_t num_unique = buildOffset();
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  const dim3 block(BlockSize);
  const dim3 grid((_num_buckets + block.x - 1) / block.x);
  switch (_comp_ratio) {
    case 1:
      CUDA_KERNEL_CALL(
          (impl::unique_kernel<IdType, 1>), grid, block, 0, stream, _bitmap, _num_buckets,
          _offset, out_row);
      break;
    case 4:
      CUDA_KERNEL_CALL(
          (impl::unique_kernel<IdType, 4>), grid, block, 0, stream, _bitmap, _num_buckets,
          _offset, out_row);
      break;
    case 8:
      CUDA_KERNEL_CALL(
          (impl::unique_kernel<IdType, 8>), grid, block, 0, stream, _bitmap, _num_buckets,
          _offset, out_row);
      break;
    case 16:
      CUDA_KERNEL_CALL(
          (impl::unique_kernel<IdType, 16>), grid, block, 0, stream, _bitmap, _num_buckets,
          _offset, out_row);
      break;
    case 32:
      CUDA_KERNEL_CALL(
          (impl::unique_kernel<IdType, 32>), grid, block, 0, stream, _bitmap, _num_buckets,
          _offset, out_row);
      break;
    default:
      LOG(ERROR) << "unsupported compression ratio, must be 1, 4, 8, 16, 32";
      break;
  }

  device->StreamSync(_ctx, stream);
  return num_unique;
};

template <typename IdType>
void DeviceBitmap::map(const IdType *row, int64_t num_rows, IdType *out_row) {
  buildOffset();
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  const dim3 block(BlockSize);
  const dim3 grid((num_rows * _comp_ratio + block.x - 1) / block.x);
  DeviceBitIterator iter(_bitmap, _num_buckets, 0);
  switch (_comp_ratio) {
    case 1:
      CUDA_KERNEL_CALL(
          (impl::map_kernel<IdType, 1>), grid, block, 0, stream, iter, _offset,
          row, num_rows, out_row);
      break;
    case 4:
      CUDA_KERNEL_CALL(
          (impl::map_kernel<IdType, 4>), grid, block, 0, stream, iter, _offset,
          row, num_rows, out_row);
      break;
    case 8:
      CUDA_KERNEL_CALL(
          (impl::map_kernel<IdType, 8>), grid, block, 0, stream, iter, _offset,
          row, num_rows, out_row);
      break;
    case 16:
      CUDA_KERNEL_CALL(
          (impl::map_kernel<IdType, 16>), grid, block, 0, stream, iter, _offset,
          row, num_rows, out_row);
      break;
    case 32:
      CUDA_KERNEL_CALL(
          (impl::map_kernel<IdType, 32>), grid, block, 0, stream, iter, _offset,
          row, num_rows, out_row);
      break;
    default:
      LOG(ERROR) << "unsupported compression ratio, must be 1, 4, 8, 16, 32";
      break;
  }
  device->StreamSync(_ctx, stream);
};

template void DeviceBitmap::flag<int32_t>(const int32_t *, int64_t);
template void DeviceBitmap::flag<int64_t>(const int64_t *, int64_t);

template int64_t DeviceBitmap::unique<int32_t>(int32_t *);
template int64_t DeviceBitmap::unique<int64_t>(int64_t *);

template void DeviceBitmap::map<int32_t>(const int32_t *, int64_t, int32_t *);
template void DeviceBitmap::map<int64_t>(const int64_t *, int64_t, int64_t *);
}  // namespace dgl::dev