//
// Created by juelinliu on 12/13/23.
//

#ifndef DGL_BITMAP_H
#define DGL_BITMAP_H
#include <dgl/array.h>

namespace dgl::dev {
class Bitmap {
 protected:
  int64_t _capacity{0};
  DGLContext _ctx;
  NDArray _bitmap;

 public:
  Bitmap() = default;
  Bitmap(int64_t capacity, DGLContext ctx);
  void Reset();
  void Mask(NDArray indices);
  int64_t NumItem();
  NDArray Flagged(NDArray rows);
};
}  // namespace dgl::dev
#endif  // DGL_BITMAP_H
