//
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
constexpr int name = 1;

#define ATEN_COMP_RATIO_SWITCH(comp_ratio, COMP_RATIO, ...) \
  do {                                                      \
    if (comp_ratio == 32) {                                 \
      constexpr int COMP_RATIO = 32;                        \
      { __VA_ARGS__ }                                       \
    } else if (comp_ratio == 16) {                          \
      constexpr int COMP_RATIO = 16;                        \
      { __VA_ARGS__ }                                       \
    } else if (comp_ratio == 8) {                           \
      constexpr int COMP_RATIO = 8;                         \
      { __VA_ARGS__ }                                       \
    } else if (comp_ratio == 4) {                           \
      constexpr int COMP_RATIO = 4;                         \
      { __VA_ARGS__ }                                       \
    } else if (comp_ratio == 2) {                           \
      constexpr int COMP_RATIO = 2;                         \
      { __VA_ARGS__ }                                       \
    } else if (comp_ratio == 1) {                           \
      constexpr int COMP_RATIO = 1;                         \
      { __VA_ARGS__ }                                       \
    } else {                                                \
      LOG(FATAL) << "Unsupported comp ratio " << comp_ratio \
                 << "; must be one of 1,2,4,8,16,32";       \
    }                                                       \
  } while (0)

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
  __device__ __forceinline__ void unflag(Distance n) {
    assert(n < _num_buckets * BucketWidth);
    const int64_t bucket_idx = (_offset + n) / BucketWidth;
    const BucketType shift = (_offset + n) % BucketWidth;
    const BucketType mask = BucketHighestBit >> shift;
    atomicXor(_bitmap + bucket_idx, mask);
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

template <typename IdType>
__global__ void unflag_kernel(
    DeviceBitIterator iter, const IdType *row, int64_t num_rows) {
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tIdx < num_rows) {
    iter.unflag(row[tIdx]);
  }
}

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

template <typename IdType, int CompRatio>
__global__ void map_kernel(
    DeviceBitIterator iter, const uint32_t advance, const OffsetType *offset,
    const IdType *row, int64_t num_rows, IdType *out_row) {
//  assert(blockDim.x == BlockSize && CompRatio <= 32);
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int32_t rIdx = tIdx / CompRatio;
  const int32_t tOffset = tIdx % CompRatio;
  typedef cub::WarpReduce<int, CompRatio> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[BlockSize / CompRatio];

  if (rIdx < num_rows) {
    const IdType id = row[rIdx];
    assert(iter[id] == true);
    const int32_t bucket_idx = (id / (BucketWidth * CompRatio)) * CompRatio;
    const int32_t total_num_bits = id % (BucketWidth * CompRatio); // excluding the [id]-th bit

    int32_t num_bits = std::max(
        0, std::min(BucketWidth, total_num_bits - tOffset * BucketWidth));

    const int bitcnt = iter.popcnt(bucket_idx + tOffset, num_bits);
    const int loc_bitcnt=bitcnt;
    const int group_id = threadIdx.x / CompRatio;
    const int agg_bitcnt = WarpReduce(temp_storage[group_id]).Sum(bitcnt);
    if (tOffset == 0) {
      const int32_t offset_idx = id / (BucketWidth * CompRatio);
      auto mapped_id = offset[offset_idx] + agg_bitcnt + advance;
      out_row[rIdx] = mapped_id;
//      if (mapped_id <= 1) printf("tIdx: %ld id: %d offset_idx: %d offset: %d loc_bits: %d agg_bits: %d num_bits: %d advance: %u out_id: %d\n", tIdx, (int)id, offset_idx, offset[offset_idx], loc_bitcnt, agg_bitcnt, num_bits, advance, mapped_id);
    }
  }
}

template <typename IdType, int CompRatio>
__global__ void map_uncheck_kernel(
    DeviceBitIterator iter, const uint32_t advance, const OffsetType *offset,
    const IdType *row, int64_t num_rows, IdType *out_row) {
  assert(blockDim.x == BlockSize && CompRatio <= 32);
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int32_t rIdx = tIdx / CompRatio;
  const int32_t tOffset = tIdx % CompRatio;
  typedef cub::WarpReduce<int, CompRatio> WarpReduce;
  __shared__
      typename WarpReduce::TempStorage temp_storage[BlockSize / CompRatio];

  if (rIdx < num_rows) {
    const IdType id = row[rIdx];
    if (iter[id]) {
      const int32_t bucket_idx = (id / (BucketWidth * CompRatio)) * CompRatio;
      const int32_t total_num_bits = id % (BucketWidth * CompRatio); // excluding the [id]-th bit

      int32_t num_bits = std::max(
          0, std::min(BucketWidth, total_num_bits - tOffset * BucketWidth));
      ;

      const int bitcnt = iter.popcnt(bucket_idx + tOffset, num_bits);
      const int group_id = threadIdx.x / CompRatio;
      const int agg_bitcnt = WarpReduce(temp_storage[group_id]).Sum(bitcnt);
      if (tOffset == 0) {
        const int32_t offset_idx = id / (BucketWidth * CompRatio);
        out_row[rIdx] = offset[offset_idx] + agg_bitcnt + advance;
      }
    }
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
    WarpScan(temp_storage[warp_id]).ExclusiveSum(cnt, cnt);
    auto out_idx = offset_start + cnt;

    while (bits != 0) {
      const int lz = __clz(bits);
      out_row[out_idx++] = BucketWidth * tIdx + lz;
      bits ^= BucketHighestBit >> lz;
    }
  }
}

}  // namespace impl

DeviceBitmap::DeviceBitmap(int64_t num_elems, DGLContext ctx, int comp_ratio) {
  _comp_ratio = comp_ratio;
  _num_buckets =
      (num_elems + BucketWidth - 1) / BucketWidth;  // 32 bits per buckets
  _num_buckets = _num_buckets + _comp_ratio -
                 _num_buckets % _comp_ratio;  // make it multiple on _comp_ratio
  _num_offset = (_num_buckets + _comp_ratio - 1) / _comp_ratio + 1;
  _ctx = ctx;
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  CUDA_CALL(cudaStreamCreateWithFlags(&_memset_stream, cudaStreamNonBlocking));
  CUDA_CALL(cudaEventCreate(&_event));

  _bitmap = static_cast<BucketType *>(device->AllocWorkspace(_ctx, _num_buckets * sizeof(BucketType)));
  CUDA_CALL(cudaMemsetAsync(_bitmap, 0, _num_buckets * sizeof(BucketType), _memset_stream));
  CUDA_CALL(cudaEventRecord(_event, _memset_stream));

  _offset = static_cast<OffsetType *>(device->AllocWorkspace(_ctx, _num_offset * sizeof(OffsetType)));
  _temp_storage_bytes = 1024 * 1024;
  _d_temp_storage = device->AllocWorkspace(_ctx, _temp_storage_bytes);
  _num_advance = 0;
  LOG(INFO) << "bitmap: " << num_elems << " on " << _ctx;
}

DeviceBitmap::~DeviceBitmap() {

  if (_event) {
    CUDA_CALL(cudaEventSynchronize(_event));
    CUDA_CALL(cudaEventDestroy(_event));
  }
  if (_memset_stream) CUDA_CALL(cudaStreamDestroy(_memset_stream));

  auto device = runtime::DeviceAPI::Get(_ctx);
  if (_bitmap) device->FreeWorkspace(_ctx, _bitmap);
  if (_offset) device->FreeWorkspace(_ctx, _offset);
  if (_d_temp_storage) device->FreeWorkspace(_ctx, _d_temp_storage);
}

void DeviceBitmap::reset() {
  //  auto device = runtime::DeviceAPI::Get(_ctx);
  _offset_built = false;
  _num_advance = 0;
  auto stream = runtime::getCurrentCUDAStream();
  CUDA_CALL(cudaStreamWaitEvent(
      stream, _event));  // must wait until all events have finished

  CUDA_CALL(cudaMemsetAsync(
      _bitmap, 0, _num_buckets * sizeof(BucketType), _memset_stream));
  CUDA_CALL(cudaEventRecord(_event, _memset_stream));
}

template <typename IdType>
void DeviceBitmap::flag(const IdType *row, int64_t num_rows) {
  auto stream = runtime::getCurrentCUDAStream();
  auto device = runtime::DeviceAPI::Get(_ctx);
  CUDA_CALL(cudaStreamWaitEvent(
      stream, _event));  // any further cuda call on runtime stream must wait until memset is completed

  const dim3 block(BlockSize);
  const dim3 grid((num_rows + block.x - 1) / block.x);
  DeviceBitIterator iter(_bitmap, _num_buckets, 0);
  CUDA_KERNEL_CALL(
      impl::flag_kernel, grid, block, 0, stream, iter, row, num_rows);
  _offset_built = false;
  CUDA_CALL(cudaEventRecord(_event, stream));
}

template <typename IdType>
void DeviceBitmap::unflag(const IdType *row, int64_t num_rows) {
  auto stream = runtime::getCurrentCUDAStream();
  auto device = runtime::DeviceAPI::Get(_ctx);
  CUDA_CALL(cudaStreamWaitEvent(
      stream, _event));  // any further cuda call on runtime stream must wait until memset is completed
  const dim3 block(BlockSize);
  const dim3 grid((num_rows + block.x - 1) / block.x);
  DeviceBitIterator iter(_bitmap, _num_buckets, 0);
  CUDA_KERNEL_CALL(
      impl::unflag_kernel, grid, block, 0, stream, iter, row, num_rows);
  _offset_built = false;
  CUDA_CALL(cudaEventRecord(_event, stream));
}

void DeviceBitmap::sync() { CUDA_CALL(cudaEventSynchronize(_event)); }

void DeviceBitmap::buildOffset() {
  if (_offset_built) return;
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  CUDA_CALL(cudaStreamWaitEvent(
      stream, _event));  // any further cuda call on runtime stream must wait until memset is completed

  DeviceBitIterator iter(_bitmap, _num_buckets, 0);

  const dim3 block(BlockSize);
  const dim3 grid((_num_buckets + block.x - 1) / block.x);
  ATEN_COMP_RATIO_SWITCH(_comp_ratio, COMP_RATIO, {
    CUDA_KERNEL_CALL(
        impl::offset_kernel<COMP_RATIO>, grid, block, 0, stream, iter,
        _num_buckets, _offset);
  });

  size_t temp_storage_bytes = 0;
  auto d_in = _offset;
  auto d_out = _offset;
  auto num_items = _num_offset;

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, temp_storage_bytes, d_in, d_out, num_items, stream));

  if (_d_temp_storage == nullptr) {
    _d_temp_storage = device->AllocWorkspace(_ctx, temp_storage_bytes);
    _temp_storage_bytes = temp_storage_bytes;
  } else if (temp_storage_bytes > _temp_storage_bytes) {
    CUDA_CALL(cudaEventSynchronize(_event));
    device->FreeWorkspace(_ctx, _d_temp_storage);
    _d_temp_storage = device->AllocWorkspace(_ctx, temp_storage_bytes);
    _temp_storage_bytes = temp_storage_bytes;
  };

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      _d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream));

  CUDA_CALL(cudaEventRecord(_event, stream));

  _offset_built = true;
}

int64_t DeviceBitmap::numItem() const {
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  //  CUDA_CALL(cudaStreamWaitEvent(stream, _event));
  CUDA_CALL(cudaStreamWaitEvent(
      stream, _event));  // any further cuda call on runtime stream must wait until memset is completed
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
      stream))
  device->StreamSync(_ctx, stream);  // don't need to record event
  return h_num_item;
}

template <typename IdType>
int64_t DeviceBitmap::unique(IdType *out_row) {
  buildOffset();
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  //  CUDA_CALL(cudaStreamWaitEvent(stream, _event));
  CUDA_CALL(cudaStreamWaitEvent(
      stream, _event));  // any further cuda call on runtime stream must wait until memset is completed
  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
        {1}, DGLDataTypeTraits<OffsetType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
        {1}, DGLDataTypeTraits<OffsetType>::dtype, DGLContext{kDGLCPU, 0});
  }
  CUDA_CALL(cudaMemcpyAsync(
      new_len_tensor.Ptr<OffsetType>(), _offset + _num_offset - 1,
      sizeof(OffsetType), cudaMemcpyDefault, stream));

  const dim3 block(BlockSize);
  const dim3 grid((_num_buckets + block.x - 1) / block.x);
  ATEN_COMP_RATIO_SWITCH(_comp_ratio, COMP_RATIO, {
    CUDA_KERNEL_CALL(
        (impl::unique_kernel<IdType, COMP_RATIO>), grid, block, 0, stream,
        _bitmap, _num_buckets, _offset, out_row);
  });

  device->StreamSync(_ctx, stream);  // don't need to record event
  _num_unique = new_len_tensor.Ptr<OffsetType>()[0];

  return _num_unique;
};

template <typename IdType>
void DeviceBitmap::map(const IdType *row, int64_t num_rows, IdType *out_row) {
  buildOffset();
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  CUDA_CALL(cudaStreamWaitEvent(stream, _event));  // any further cuda call on runtime stream must wait until memset is completed
  const dim3 block(BlockSize);
  const dim3 grid((num_rows * _comp_ratio + block.x - 1) / block.x);
  DeviceBitIterator iter(_bitmap, _num_buckets, 0);
  ATEN_COMP_RATIO_SWITCH(_comp_ratio, COMP_RATIO, {
    CUDA_KERNEL_CALL(
        (impl::map_kernel<IdType, COMP_RATIO>), grid, block, 0, stream, iter,
        _num_advance, _offset, row, num_rows, out_row);
  });
  CHECK_EQ(_num_advance, 0);
  //  device->StreamSync(_ctx, stream);
  CUDA_CALL(cudaEventRecord(_event, stream));
};

template <typename IdType>
void DeviceBitmap::map_uncheck(const IdType *row, int64_t num_rows, IdType *out_row) {
  buildOffset();
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
    CUDA_CALL(cudaStreamWaitEvent(stream, _event));

  const dim3 block(BlockSize);
  const dim3 grid((num_rows * _comp_ratio + block.x - 1) / block.x);
  DeviceBitIterator iter(_bitmap, _num_buckets, 0);
  ATEN_COMP_RATIO_SWITCH(_comp_ratio, COMP_RATIO, {
    CUDA_KERNEL_CALL(
        (impl::map_uncheck_kernel<IdType, COMP_RATIO>), grid, block, 0, stream, iter,
        _num_advance, _offset, row, num_rows, out_row);
  });
  //  device->StreamSync(_ctx, stream);
  CUDA_CALL(cudaEventRecord(_event, stream));
};

template void DeviceBitmap::flag<int32_t>(const int32_t *, int64_t);
template void DeviceBitmap::flag<int64_t>(const int64_t *, int64_t);

template void DeviceBitmap::unflag<int32_t>(const int32_t *, int64_t);
template void DeviceBitmap::unflag<int64_t>(const int64_t *, int64_t);

template int64_t DeviceBitmap::unique<int32_t>(int32_t *);
template int64_t DeviceBitmap::unique<int64_t>(int64_t *);

template void DeviceBitmap::map<int32_t>(const int32_t *, int64_t, int32_t *);
template void DeviceBitmap::map<int64_t>(const int64_t *, int64_t, int64_t *);

template void DeviceBitmap::map_uncheck<int32_t>(const int32_t *, int64_t, int32_t *);
template void DeviceBitmap::map_uncheck<int64_t>(const int64_t *, int64_t, int64_t *);
}  // namespace dgl::dev