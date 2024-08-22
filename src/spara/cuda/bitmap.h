//
// Created by juelin on 2/6/24.
//

#ifndef DGL_BITMAP_H
#define DGL_BITMAP_H

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include <cstdint>

namespace dgl::dev {

typedef int32_t BucketType;
typedef uint32_t OffsetType;

struct QueryIdx
{
  NDArray _hitReadIdx;
  NDArray _hitWriteIdx;
  NDArray _missReadId;
  NDArray _missWriteIdx;
};
class DeviceBitmap {
 private:
  BucketType* _bitmap{nullptr};  // 1 if mapped 0 otherwise
  OffsetType* _offset{nullptr};  // used for fast indexing during mapping
  void* _d_temp_storage{nullptr};// temp storage for prefix sum
  DGLContext _ctx{};
  cudaEvent_t _event{};
  cudaStream_t _memset_stream{}; // a separate stream is used to set the bitmap to zero
  uint32_t _num_elems{0};
  uint32_t _num_buckets{0};
  uint32_t _num_offset{0};
  uint32_t _comp_ratio{8};  // number of 32-bit buckets counted per offset
  uint32_t _num_unique{0};
  uint32_t _num_advance{0}; // all the remapped index will be advanced by _num_advance
  uint32_t _temp_storage_bytes{0};
  bool _offset_built{false};
  void* get_temp_storage(size_t temp_storage_bytes);
 public:
  DeviceBitmap(int64_t num_elems, DGLContext ctx, int _comp_ratio = 8);
  ~DeviceBitmap();
  /**
   *
   * @return number of 1 bits in the bitmap
   */
  int64_t numItem() const; // sync call

  void set_advance(uint32_t advance){_num_advance = advance;};

  void sync(); // sync call
  void reset(); // async
  /**
      @params: IdType: type of the input row
      @params: d_row: device array with indices to be mapped as 1
      @params: num_rows: number of elements in row
      return total number of bits mapped as 1
  */
  template <typename IdType>
  void flag(const IdType* d_row, int64_t num_rows);  // async

//  template <typename IdType>
//  void unflag(const IdType* d_row, int64_t num_rows);  // async
  /**
      create bitmap global to local id maps
  */
  void buildOffset(); // async

  /**
      @params: IdType: type of the output row
      @params: out_row: device buffer array to store the indices of bits flagged
     as 1
  */
  template <typename IdType>
  int64_t unique(IdType* d_out_row); // sync call

  /**
      @params: IdType: type of the input and output row
      @params: d_row: device input row, assume all ids in it has been flagged
      @params: num_rows: lenght of input (and output) row
      @params: out_row: output buffer
      @params: write the indices of bits flagged as 1 in out_row
  */
  template <typename IdType>
  void map(const IdType* d_row, int64_t num_rows, IdType* d_out_row); // async

//  /**
//    @params: IdType: type of the input and output row
//    @params: d_row: device input row, assume all ids in it has been flagged
//    @params: num_rows: lenght of input (and output) row
//    @params: out_row: output buffer
//    @params: write the indices of bits flagged as 1 in out_row
//*/
//  template <typename IdType>
//  void map_uncheck(const IdType* d_row, int64_t num_rows, IdType* d_out_row); // async

  /**
  @params: flag: 1 if the corresponding vid is cached
  @params: input_node: the input nodes to be queries
*/
  template <typename IdType>
  QueryIdx queryBitmap(const IdType* d_row, int64_t num_rows);

};  // DeviceBitmap

std::shared_ptr<DeviceBitmap> getStaticBitmap(int64_t num_elems, DGLContext ctx, int comp_ratio = 8);

}  // namespace dgl::dev
#endif  // DGL_BITMAP_H