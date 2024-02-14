#ifndef DGL_GROOT_ARRAY_SCATTER_H_
#define DGL_GROOT_ARRAY_SCATTER_H_

#include <dgl/array.h>
#include <dgl/packed_func_ext.h>

namespace dgl::dev {
class ScatteredArray;

void Scatter(
    int64_t rank, int64_t world_size, int64_t num_partitions,
    const NDArray& frontier, const NDArray& _partition_map,
    ScatteredArray array);

class ScatteredArrayObject : public runtime::Object {
 public:
  int64_t _scatter_dim{0};
  int64_t _unique_dim{0};
  int64_t _expect_size{0};
  DGLDataType dtype;
  DGLContext ctx;
  bool debug = false;
  // original array has no duplicates
  // original array is partition discontinuous
  NDArray originalArray;
  // Partition map does not refer to global IDS, this is local here so that when
  // we move blocks, partition map compute for dest vertices can be reused
  NDArray partitionMap;

  // original array is scattered such that partitions are contiguous, resulting
  // in partitionContinuos array
  NDArray partitionContinuousArray;
  NDArray partitionContinuousOffsets;
  // Idx are only needed when moving Vertices, edges dont require this.
  NDArray gather_idx_in_part_disc_cont;   // shape of original array
  NDArray scatter_idx_in_part_disc_cont;  // shape of scattered array

  // After NCCL Comm
  NDArray shuffled_array;
  NDArray shuffled_recv_offsets;

  //  Possible received array after shuffling has duplicates
  //  std::shared_ptr<CudaHashTable> table;
  NDArray unique_array;
  NDArray gather_idx_in_unique_out_shuffled;

  ~ScatteredArrayObject() = default;
  ScatteredArrayObject(
      int64_t expectedSize, DGLContext ctx, DGLDataType dtype) {
    this->dtype = dtype;
    this->ctx = ctx;
    this->_expect_size = expectedSize;
    LOG(INFO) << "Scatter array expected size " << expectedSize / 1024 / 1024 << " M";
  }


  NDArray shuffle_forward(const NDArray& feat, int rank, int world_size) const;

  NDArray shuffle_backward(const NDArray& back_grad, int rank, int world_size) const;

  void VisitAttrs(runtime::AttrVisitor *v) final {
    v->Visit("original_array", &originalArray);
    v->Visit("partition_map", &partitionMap);
    v->Visit("partitionContinuousArray", &partitionContinuousArray);
    v->Visit("gather_idx_in_part_disc_cont", &gather_idx_in_part_disc_cont);
    v->Visit("scatter_idx_in_part_disc_cont", &scatter_idx_in_part_disc_cont);
    v->Visit("partition_continuous_offsets", &partitionContinuousOffsets);
    v->Visit("unique_array", &unique_array);
    v->Visit(
        "gather_idx_in_unique_out_shuffled",
        &gather_idx_in_unique_out_shuffled);
  }

  static constexpr const char *_type_key = "ScatteredArray";
  DGL_DECLARE_OBJECT_TYPE_INFO(ScatteredArrayObject, Object);
};

class ScatteredArray : public runtime::ObjectRef {
 public:
  DGL_DEFINE_OBJECT_REF_METHODS(
      ScatteredArray, runtime::ObjectRef, ScatteredArrayObject);
  static ScatteredArray Create(
      int64_t expected_size_on_single_gpu,
      int64_t num_partitions,
      DGLContext ctx, DGLDataType dtype) {

    return ScatteredArray(std::make_shared<ScatteredArrayObject>(
        expected_size_on_single_gpu * num_partitions * num_partitions, ctx, dtype));
  }
};

}  // namespace dgl::groot

#endif  // DGL_GROOT_ARRAY_SCATTER_H_