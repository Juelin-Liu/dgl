#ifndef DGL_GROOT_ARRAY_SCATTER_H_
#define DGL_GROOT_ARRAY_SCATTER_H_

#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <nccl.h>
namespace dgl::dev {
class ScatteredArray;

void Scatter(
    int64_t rank, int64_t world_size, int64_t num_partitions,
    const NDArray& local_unique_src, const NDArray& local_partition_idx,
    ScatteredArray array);

class ScatteredArrayObject : public runtime::Object {
 public:
  int64_t _scatter_dim{0};
  int64_t _unique_dim{0};
  int64_t _v_num{0};

  DGLDataType dtype;
  DGLContext ctx;
  bool debug = false;
  ncclComm_t _nccl_comm{nullptr};
  // original array has no duplicates
  // original array is partition discontinuous
  NDArray local_unique_src;
  // Partition map does not refer to global IDS, this is local here so that when
  // we move blocks, partition map compute for dest vertices can be reused
  NDArray local_part_idx;

  // original array is scattered such that partitions are contiguous, resulting
  // in partitionContinuos array
  NDArray send_offset;
  NDArray local_unique_src_offset;
  // Idx are only needed when moving Vertices, edges dont require this.
  NDArray gather_idx;   // shape of original array
  NDArray scatter_idx;  // shape of scattered array

  // After NCCL Comm
  NDArray global_src;
  NDArray global_src_offset;

  //  Possible received array after shuffling has duplicates
  //  std::shared_ptr<CudaHashTable> table;
  NDArray unique_array;
  NDArray gather_idx_in_unique_out_shuffled;

  ~ScatteredArrayObject() = default;
  ScatteredArrayObject(
      int64_t v_num, DGLContext ctx, DGLDataType dtype, ncclComm_t nccl_comm) {
    this->dtype = dtype;
    this->ctx = ctx;
    this->_nccl_comm = nccl_comm;
    this->_v_num = v_num;
  }


  NDArray shuffle_forward(const NDArray& feat, int rank, int world_size) const;

  NDArray shuffle_backward(const NDArray& back_grad, int rank, int world_size) const;

  void VisitAttrs(runtime::AttrVisitor *v) final {
    v->Visit("original_array", &local_unique_src);
    v->Visit("partition_map", &local_part_idx);
    v->Visit("partitionContinuousArray", &send_offset);
    v->Visit("gather_idx_in_part_disc_cont", &gather_idx);
    v->Visit("scatter_idx_in_part_disc_cont", &scatter_idx);
    v->Visit("partition_continuous_offsets", &local_unique_src_offset);
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
      int64_t v_num, DGLContext ctx, DGLDataType dtype, ncclComm_t nccl_comm) {

    return ScatteredArray(std::make_shared<ScatteredArrayObject>(
        v_num, ctx, dtype, nccl_comm));
  }
};

}  // namespace dgl::dev

#endif  // DGL_GROOT_ARRAY_SCATTER_H_