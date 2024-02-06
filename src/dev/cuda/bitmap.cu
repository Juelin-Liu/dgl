//
// Created by juelin on 2/6/24.
//
#include <cub/cub.cuh>

#include "../../runtime/cuda/cuda_common.h"
#include "bitmap.h"

namespace dgl::dev {
class DeviceBitIterator
    : public std::iterator<std::random_access_iterator_tag, bool> {
 private:
  Bucket* _bitmap{nullptr};
  int64_t _offset{0};
  int64_t _num_buckets{0};

 public:
  using self_type = DeviceBitIterator;
  using value_type = bool;

  __host__ __device__
  DeviceBitIterator(Bucket* bitmap, int64_t num_buckets, int64_t offset = 0)
      : _bitmap{bitmap},
        _offset{offset},
        _num_buckets{num_buckets} {
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
    const uint32_t bucket_idx = _offset / (sizeof(Bucket) * 8);
    const Bucket shift = _offset % (sizeof(Bucket) * 8);
    const Bucket mask = 1u << shift;
    const Bucket flag = _bitmap[bucket_idx];
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
    const Distance bucket_idx = (_offset + n) / (sizeof(Bucket) * 8);
    const Bucket shift = (_offset + n) % (sizeof(Bucket) * 8);
    const Bucket mask = 1u << shift;
    const Bucket flag = _bitmap[bucket_idx];
    bool retval = (flag & mask);
    // printf("operator[] n %d flag %d shift %d mask %d offset %ld ret %d\n", n,
    // flag, shift, mask, _offset, retval);
    return retval;
  }

  template <typename Distance>
  __device__ __forceinline__ void flag(Distance n) {
    assert(n < _num_buckets * sizeof(Bucket) * 8);
    const Distance bucket_idx = (_offset + n) / (sizeof(Bucket) * 8);
    const Bucket shift = (_offset + n) % (sizeof(Bucket) * 8);
    const Bucket mask = 1u << shift;
    atomicOr(_bitmap + bucket_idx, mask);
    // printf("flag n %ld shift %d mask %d offset %ld bitmap %d\n", n, shift,
    // mask, _offset, _bitmap[bucket_idx]);
  }

  template <typename Distance>
  __device__ __forceinline__ Bucket popcnt(Distance bucket_idx) const {
    return __popc(_bitmap[bucket_idx]);
  }

  template <typename Distance>
  __device__ __forceinline__ Bucket
  popcnt(Distance bucket_idx, Distance num_bits) const {
    const int shift = 8 * sizeof(Bucket) - num_bits;
    const Bucket bitmap = _bitmap[bucket_idx];
    const Bucket mask = bitmap << shift;
    const int retval = __popc(mask);
    // printf("popcnt bucket_idx %ld num_bits %ld retval %d bitmap %d shift %d
    // mask %d\n", bucket_idx, num_bits, retval, bitmap, shift, mask);
    return retval;
  }
  __host__ __device__ __forceinline__ bool operator==(
      const self_type& rhs) const {
    return _bitmap == rhs._bitmap && _offset == rhs._offset &&
           _num_buckets == rhs._num_buckets;
  }

  __host__ __device__ __forceinline__ bool operator!=(
      const self_type& rhs) const {
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

// assume d_num_item_out is 0
__global__ void cnt_kernel(DeviceBitIterator iter, int64_t num_buckets, uint32_t * d_num_item_out) {
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  typedef cub::BlockReduce<uint32_t, 256> BlockReduce;
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

__global__ void popcnt_kernel(
    DeviceBitIterator iter, int64_t num_buckets, Offset *offset) {
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tIdx < num_buckets) {
    offset[tIdx] = iter.popcnt(tIdx);
  }
}

template <typename IdType>
__global__ void map_kernel(
    DeviceBitIterator iter, const Offset *offset, const IdType *row,
    int64_t num_rows, IdType *out_row) {
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tIdx < num_rows) {
    const IdType id = row[tIdx];
    assert(iter[id] == true);
    const int64_t bucket_idx = id / (8 * sizeof(Bucket));
    const int64_t num_bits = id % (8 * sizeof(Bucket)) + 1;
    const IdType loc_id =
        offset[bucket_idx] + iter.popcnt(bucket_idx, num_bits);
    out_row[tIdx] = loc_id;
  }
}

__global__ void read_kernel(DeviceBitIterator iter, int64_t num_items) {
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tIdx < num_items) {
    const int flag = iter[tIdx];
    printf("tIdx: %lld mask %d\n", tIdx, flag);
  }
}
}  // namespace impl

DeviceBitmap::DeviceBitmap(int64_t num_elems, DGLContext ctx, bool allow_remap) {
  _allow_remap = allow_remap;
  int64_t bucket_bits = sizeof(Bucket) * 8;
  _num_buckets = (num_elems + bucket_bits - 1) / bucket_bits;  // 32 bits per buckets
  _ctx = ctx;
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();

  _bitmap = static_cast<Bucket *>(device->AllocWorkspace(ctx, _num_buckets * sizeof(Bucket)));
  CUDA_CALL(cudaMemsetAsync(&_bitmap, 0, _num_buckets * sizeof(Bucket), stream));
  if (_allow_remap) {
    _offset = static_cast<Bucket *>(device->AllocWorkspace(ctx, (_num_buckets + 1) * sizeof(Offset)));
    CUDA_CALL(cudaMemsetAsync(&_offset, 0, (_num_buckets + 1) * sizeof(Bucket), stream));
  }
}

DeviceBitmap::~DeviceBitmap() {
//  if (_bitmap) cudaFree(_bitmap);
//  if (_offset) cudaFree(_offset);
  auto device = runtime::DeviceAPI::Get(_ctx);

  if (_bitmap) device->FreeWorkspace(_ctx, _bitmap);
  if (_offset) device->FreeWorkspace(_ctx, _offset);
}

void DeviceBitmap::reset() {
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  cudaMemsetAsync(&_bitmap, 0, _num_buckets * sizeof(Bucket), stream);
  if(_allow_remap) cudaMemsetAsync(&_offset, 0, (_num_buckets + 1) * sizeof(Bucket), stream);
  device->StreamSync(_ctx, stream);
  _num_flagged = 0;
}

template <typename IdType>
void DeviceBitmap::flag(const IdType *row, int64_t num_rows) {
  auto stream = runtime::getCurrentCUDAStream();
  auto device = runtime::DeviceAPI::Get(_ctx);

  const dim3 block(256);
  const dim3 grid((num_rows + block.x - 1) / block.x);
  DeviceBitIterator iter(_bitmap, _num_buckets, 0);
  CUDA_KERNEL_CALL(impl::flag_kernel, grid, block, 0, stream, iter, row, num_rows);
//  impl::flag_kernel<<<grid, block, 0>>>(iter, row, num_rows);
//  device->StreamSync(_ctx, stream);
  _build_map = false;
}


int64_t DeviceBitmap::buildMap() {
  CHECK(_allow_remap);
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  const dim3 block(256);
  const dim3 grid((_num_buckets + block.x - 1) / block.x);
  DeviceBitIterator iter(_bitmap, _num_buckets, 0);
  CUDA_KERNEL_CALL(impl::popcnt_kernel, grid, block, 0, stream, iter, _num_buckets, _offset);
//  impl::popcnt_kernel<<<grid, block, 0>>>(iter, _num_buckets, _offset);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  auto d_in = _offset;
  auto d_out = _offset;
  auto num_items = _num_buckets + 1;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream));
//  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  d_temp_storage = device->AllocWorkspace(_ctx, temp_storage_bytes);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream));
  CUDA_CALL(cudaMemcpyAsync(
      &_num_flagged, _offset + _num_buckets, sizeof(Offset), cudaMemcpyDefault, stream));

  device->StreamSync(_ctx, stream);
  device->FreeWorkspace(_ctx, d_temp_storage);
  _build_map = true;
  return _num_flagged;
}

int64_t DeviceBitmap::numItem() const {
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  const dim3 block(256);
  const dim3 grid((_num_buckets + block.x - 1) / block.x);
  DeviceBitIterator iter(_bitmap, _num_buckets, 0);
  auto * d_num_item = static_cast<uint32_t*>(device->AllocWorkspace(_ctx, sizeof(uint32_t)));
  uint32_t h_num_item{0};
  CUDA_CALL(cudaMemsetAsync(d_num_item, 0, sizeof(uint32_t), stream));
  CUDA_KERNEL_CALL(impl::cnt_kernel, grid, block, 0, stream, iter, _num_buckets, d_num_item);
  CUDA_CALL(cudaMemcpyAsync(&h_num_item, d_num_item, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  device->StreamSync(_ctx, stream);
  return h_num_item;
}

template <typename IdType>
int64_t DeviceBitmap::unique(IdType *out_row) const {
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  // adopted from cub
  // Declare, allocate, and initialize device-accessible pointers for input,
  // flags, and output
  int num_items = _num_buckets * sizeof(Bucket) * 8;
  auto d_in =
      cub::CountingInputIterator<IdType>(0);  // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
  DeviceBitIterator d_flags(
      _bitmap, _num_buckets, 0);  // e.g., [1, 0, 0, 1, 0, 1, 1, 0]
  IdType *d_out = out_row;        // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  int64_t *d_num_selected_out = static_cast<int64_t *>(device->AllocWorkspace(_ctx, sizeof(int64_t)));
//  cudaMalloc(&d_num_selected_out, sizeof(int64_t));
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CUDA_CALL(cub::DeviceSelect::Flagged(
      d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out,
      d_num_selected_out, num_items, stream));
  device->StreamSync(_ctx, stream);
  // Allocate temporary storage
//  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  d_temp_storage = device->AllocWorkspace(_ctx, temp_storage_bytes);
  // Run selection
  CUDA_CALL(cub::DeviceSelect::Flagged(
      d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out,
      d_num_selected_out, num_items, stream));
  // d_out                 <-- [1, 4, 6, 7]
  // d_num_selected_out    <-- [4]
  int64_t h_num_selected_out{0};
  cudaDeviceSynchronize();
  CUDA_CALL(cudaMemcpyAsync(
      &h_num_selected_out, d_num_selected_out, sizeof(int64_t),
      cudaMemcpyDefault, stream));
//  cudaFree(d_num_selected_out);
//  cudaDeviceSynchronize();
  device->StreamSync(_ctx, stream);
  assert(h_num_selected_out == _num_flagged);
  device->FreeWorkspace(_ctx, d_temp_storage);
  device->FreeWorkspace(_ctx, d_num_selected_out);
  return h_num_selected_out;
};

template <typename IdType>
void DeviceBitmap::map(const IdType *row, int64_t num_rows, IdType *out_row) const {
  CHECK(_allow_remap && _build_map);
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto stream = runtime::getCurrentCUDAStream();
  const dim3 block(256);
  const dim3 grid((num_rows + block.x - 1) / block.x);
  DeviceBitIterator iter(_bitmap, _num_buckets, 0);
  CUDA_KERNEL_CALL(impl::map_kernel, grid, block, 0, stream, iter, _offset, row, num_rows, out_row);
//  impl::map_kernel<<<grid, block, 0>>>(iter, _offset, row, num_rows, out_row);
//  cudaDeviceSynchronize();
  device->StreamSync(_ctx, stream);
};

template void DeviceBitmap::flag<int32_t>(const int32_t *, int64_t);
template void DeviceBitmap::flag<int64_t>(const int64_t *, int64_t);

template int64_t DeviceBitmap::unique<int32_t>(int32_t *) const ;
template int64_t DeviceBitmap::unique<int64_t>(int64_t *) const ;

template void DeviceBitmap::map<int32_t>(const int32_t *, int64_t, int32_t *) const ;
template void DeviceBitmap::map<int64_t>(const int64_t *, int64_t, int64_t *) const ;
}  // namespace dgl::dev