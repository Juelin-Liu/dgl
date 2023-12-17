//
// Created by juelinliu on 12/13/23.
//

#ifndef DGL_BITMAP_H
#define DGL_BITMAP_H
#include <dgl/array.h>
#include "array_ops.cuh"

namespace dgl::dev {
class Bitmap {
 protected:
  int64_t _capacity{0};
  DGLContext _ctx;
  NDArray _bitmap;

 public:
  Bitmap(int64_t capacity, DGLContext ctx) {
    _capacity = capacity;
    _ctx = ctx;
    _bitmap = NDArray::Empty({_capacity}, DGLDataTypeTraits<int8_t>::dtype, _ctx);
    Reset();
  }

  NDArray Flagged(DGLDataType idtype) {
    ATEN_ID_TYPE_SWITCH(idtype, IdType, {
      return dev::Flagged<kDGLCUDA, IdType>(_bitmap, _ctx);
    });
  }

  void Reset() { return dev::Reset(_bitmap); };

  void Mask(const NDArray& indices) {
    ATEN_ID_TYPE_SWITCH(indices->dtype, IdType, {
      return dev::Mask<kDGLCUDA, IdType>(_bitmap, indices);
    });
  };

  int64_t NumItem() { return dev::NumItem(_bitmap); };
};
}  // namespace dgl::dev
#endif  // DGL_BITMAP_H
