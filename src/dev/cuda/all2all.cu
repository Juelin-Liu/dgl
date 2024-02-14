//
// Created by juelin on 2/14/24.
//

#include "./all2all.h"
#include "../../runtime/cuda/cuda_common.h"

namespace dgl::dev {

template<typename IdType>
__global__ void DiffKernel(IdType* out, const IdType* in, int64_t size) {
  int tid = threadIdx.x;
  if (tid < size) {
    out[tid] = in[tid + 1] - in[tid];
  }
}

template __global__ void DiffKernel<int32_t >(int32_t *, const int32_t *, int64_t);
template __global__ void DiffKernel<int64_t >(int64_t *, const int64_t *, int64_t);

IdArray Diff(const IdArray& prefix_sum) {
  auto stream = runtime::getCurrentCUDAStream();
  int64_t size = prefix_sum->shape[0] - 1;
  IdArray ret = IdArray::Empty({size}, prefix_sum->dtype, prefix_sum->ctx);
  ATEN_ID_TYPE_SWITCH(ret->dtype, IdType, {
    CUDA_KERNEL_CALL(
        DiffKernel, 1, 256, 0, stream, ret.Ptr<IdType>(),
        prefix_sum.Ptr<IdType>(), size);
  });
  return ret;
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void NCCLAllToAll(
    IdArray send_buffer, const IdArray& send_offset,
    IdArray recv_buffer, const IdArray& recv_offset,
    int expand_size, int rank, int world_size,
    ncclComm_t nccl_comm, cudaStream_t stream) {
  //  auto stream = CUDAThreadEntry::ThreadLocal()->stream;
  //  auto data_copy_stream = CUDAThreadEntry::ThreadLocal()->data_copy_stream;
  T* send_buffer_ptr = send_buffer.Ptr<T>();
  T* recv_buffer_ptr = recv_buffer.Ptr<T>();
  int type_bytes = sizeof(T);
  ATEN_ID_TYPE_SWITCH(send_offset->dtype, IdType, {
    auto* send_offset_ptr = send_offset.Ptr<IdType>();
    auto* recv_offset_ptr = recv_offset.Ptr<IdType>();
    ncclGroupStart();
    for (int r = 0; r < world_size; ++r) {
      if (r != rank) {
        IdType send_size =
            (send_offset_ptr[r + 1] - send_offset_ptr[r]) * expand_size;
        IdType send_ptr = send_offset_ptr[r] * expand_size;
        IdType recv_size =
            (recv_offset_ptr[r + 1] - recv_offset_ptr[r]) * expand_size;
        IdType recv_ptr = recv_offset_ptr[r] * expand_size;
        ncclSend(
            send_buffer_ptr + send_ptr, send_size, NCCL_DATA_TYPE, r, nccl_comm,
            stream);
        ncclRecv(
            recv_buffer_ptr + recv_ptr, recv_size, NCCL_DATA_TYPE, r, nccl_comm,
            stream);
      }
    }
    ncclGroupEnd();

    CUDA_CALL(cudaMemcpyAsync(
        recv_buffer_ptr + recv_offset_ptr[rank] * expand_size,
        send_buffer_ptr + send_offset_ptr[rank] * expand_size,
        (send_offset_ptr[rank + 1] - send_offset_ptr[rank]) * expand_size *
            type_bytes,
        cudaMemcpyDeviceToDevice, stream));
//    CUDA_CALL(cudaStreamSynchronize(stream));
  });
}

std::pair<IdArray, IdArray> Alltoall(
    int64_t rank, int64_t world_size, const IdArray& input, int64_t expand_size,
    const IdArray& send_offset, IdArray recv_offset, cudaStream_t stream){
  // NCCL
  auto nccl_comm = getNccl();
  CHECK_NE(nccl_comm, nullptr) << "Must call initNccl after spawn the process";

  ATEN_ID_TYPE_SWITCH(input->dtype, IdType, {
//  CHECK(send_offset->dtype.bits == 64);
  auto send_sizes = Diff(send_offset);
  auto ctx = input->ctx;
  auto dtype = input->dtype;
  auto host_ctx = DGLContext{kDGLCPU, 0};
//  auto nccl_comm = getNcclCommunicator(rank);
  // NOTE: to guarantee the send_offset is ready
//  CUDA_CALL(cudaStreamSynchronize(stream));
  // it is captured in the stream when you use copy to
  CHECK(send_offset->data != nullptr);
  auto host_send_offset = static_cast<NDArray>(send_offset).PinMemory();
  CUDA_CALL(cudaStreamSynchronize(stream));
  if (aten::IsNullArray(recv_offset)) {
    IdArray recv_sizes =
        IdArray::Empty({world_size}, send_offset->dtype, ctx);
    IdArray range_seq = aten::Range(0, world_size + 1, 64, host_ctx);
    if (send_offset->dtype.bits == 32) {
      NCCLAllToAll<int32_t, ncclInt32>(
          send_sizes, range_seq, recv_sizes, range_seq, 1, rank, world_size,
          nccl_comm, stream);
    } else {
      NCCLAllToAll<int64_t, ncclInt64>(
          send_sizes, range_seq, recv_sizes, range_seq, 1, rank, world_size,
          nccl_comm, stream);
    }
    CUDA_CALL(cudaStreamSynchronize(stream));
    recv_offset = aten::CumSum(recv_sizes, true);
  }

  auto host_recv_offset = recv_offset.PinMemory();
  CUDA_CALL(cudaStreamSynchronize(stream));
    auto* host_recv_offset_ptr = host_recv_offset.Ptr<IdType>();
    int n_recv = host_recv_offset_ptr[world_size] * expand_size;
    auto recvbuff = IdArray::Empty({n_recv}, input->dtype, ctx);
    //   scheduler->TryComm(thread_id);
    if (input->dtype.code == 0) {
      if (input->dtype.bits == 32) {
        NCCLAllToAll<int, ncclInt32>(
            input, host_send_offset, recvbuff, host_recv_offset, expand_size,
            rank, world_size, nccl_comm, stream);
      } else {
        NCCLAllToAll<int64_t, ncclInt64>(
            input, host_send_offset, recvbuff, host_recv_offset, expand_size,
            rank, world_size, nccl_comm, stream);
      }
    } else {
      if (input->dtype.bits == 32) {
        NCCLAllToAll<float, ncclFloat32>(
            input, host_send_offset, recvbuff, host_recv_offset, expand_size,
            rank, world_size, nccl_comm, stream);
      } else {
        NCCLAllToAll<double, ncclFloat64>(
            input, host_send_offset, recvbuff, host_recv_offset, expand_size,
            rank, world_size, nccl_comm, stream);
      }
    };
    CUDA_CALL(cudaStreamSynchronize(stream));
    return {recvbuff, recv_offset};});
}
}  // namespace dgl::ds