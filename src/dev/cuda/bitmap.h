//
// Created by juelin on 2/6/24.
//

#ifndef DGL_BITMAP_H
#define DGL_BITMAP_H

#include <cstdint>
#include <iostream>
#include <iterator>
#include <cassert>
#include <dgl/runtime/device_api.h>

namespace dgl::dev {
typedef uint32_t Bucket;
typedef uint32_t Offset;
class DeviceBitmap {
 private:
  Bucket* _bitmap{nullptr};  // 1 if mapped 0 otherwise
  Offset* _offset{nullptr};  // used for fast indexing during mapping
  DGLContext _ctx{};
  int64_t _num_flagged{0};
  int64_t _num_buckets{0};
  bool _allow_remap{true};
  bool _build_map{false};
 public:
  DeviceBitmap(int64_t num_elems, DGLContext ctx, bool allow_remap);
  ~DeviceBitmap();

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
  int64_t buildMap();

  /**
   *
   * @return number of 1 bits in the bitmap
   */
  int64_t numItem() const;

  /**
      @params: IdType: type of the output row
      @params: out_row: device buffer array to store the indices of bits flagged
     as 1
  */
  template <typename IdType>
  int64_t unique(IdType* d_out_row) const;

  /**
      @params: IdType: type of the input and output row
      @params: d_row: device input row, assume all ids in it has been flagged
      @params: num_rows: lenght of input (and output) row
      @params: out_row: output buffer
      @params: write the indices of bits flagged as 1 in out_row
  */
  template <typename IdType>
  void map(const IdType* d_row, int64_t num_rows, IdType* d_out_row) const;
};  // DeviceBitmap
}  // namespace dgl::dev
#endif  // DGL_BITMAP_H
