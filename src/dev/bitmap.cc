//
// Created by juelinliu on 12/13/23.
//

#include "bitmap.h"

#include "cuda/array_ops.cuh"

namespace dgl::dev {
Bitmap::Bitmap(int64_t capacity, DGLContext ctx) {
  int64_t arr_len = (capacity + sizeof(bool) - 1) / 8;
  _capacity = capacity;
  _ctx = ctx;
  _bitmap = NDArray::Empty({arr_len}, DGLDataTypeTraits<bool>::dtype, _ctx);
  Reset();
}

NDArray Bitmap::Flagged(NDArray rows) {
  ATEN_ID_TYPE_SWITCH(rows->dtype, IdType, {
    return dev::Flagged<kDGLCUDA, IdType>(_bitmap, _capacity, _ctx);
  });
}

void Bitmap::Reset() { return dev::Reset(_bitmap, _capacity); };

void Bitmap::Mask(NDArray indices) {
  ATEN_ID_TYPE_SWITCH(indices->dtype, IdType, {
    return dev::Mask<kDGLCUDA, IdType>(_bitmap, _capacity, indices);
  });
};

int64_t Bitmap::NumItem() { return dev::NumItem(_bitmap, _capacity); };
}  // namespace dgl::dev
