//
// Created by juelin on 2/6/24.
//

#ifndef DGL_BITMAP_H
#define DGL_BITMAP_H

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>

namespace dgl::dev {
typedef int32_t BucketType;
typedef uint32_t OffsetType;
class DeviceBitmap {
 private:
  BucketType* _bitmap{nullptr};  // 1 if mapped 0 otherwise
  OffsetType* _offset{nullptr};  // used for fast indexing during mapping
  void* _d_temp_storage{nullptr};// temp storage for prefix sum
  DGLContext _ctx{};
  cudaEvent_t _event{};
  uint32_t _num_buckets{0};
  uint32_t _num_offset{0};
  uint32_t _comp_ratio{4};  // number of 32-bit buckets counted per offset
  uint32_t _num_unique{0};
  uint32_t _temp_storage_bytes{0};
  bool _offset_built{false};

 public:
  DeviceBitmap(int64_t num_elems, DGLContext ctx, int _comp_ratio = 8);
  ~DeviceBitmap();
  /**
   *
   * @return number of 1 bits in the bitmap
   */
  int64_t numItem() const;

  void reset();
  /**
      @params: IdType: type of the input row
      @params: d_row: device array with indices to be mapped as 1
      @params: num_rows: number of elements in row
      return total number of bits mapped as 1
  */
  template <typename IdType>
  void flag(const IdType* d_row, int64_t num_rows);  // return number of flagged

  /**
      create bitmap global to local id maps
  */
  int64_t buildOffset();

  /**
      @params: IdType: type of the output row
      @params: out_row: device buffer array to store the indices of bits flagged
     as 1
  */
  template <typename IdType>
  int64_t unique(IdType* d_out_row);

  /**
      @params: IdType: type of the input and output row
      @params: d_row: device input row, assume all ids in it has been flagged
      @params: num_rows: lenght of input (and output) row
      @params: out_row: output buffer
      @params: write the indices of bits flagged as 1 in out_row
  */
  template <typename IdType>
  void map(const IdType* d_row, int64_t num_rows, IdType* d_out_row);
};  // DeviceBitmap
}  // namespace dgl::dev
#endif  // DGL_BITMAP_H
