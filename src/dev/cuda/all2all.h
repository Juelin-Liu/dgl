//
// Created by juelin on 2/14/24.
//

#ifndef DGL_ALL2ALL_H
#define DGL_ALL2ALL_H

#include <dgl/array.h>
#include <nccl.h>
#include <memory>

namespace dgl::dev {


static std::shared_ptr<ncclComm_t> getNcclPtr() {
  static auto comm = std::make_shared<ncclComm_t>();
  return comm;
}

static ncclComm_t getNccl() {
    return *getNcclPtr().get();
}
/**
 * @brief Unified alltoall communication.
 *
 * @param input input on GPU
 * @param send_offset send_offset on GPU
 * @param rank
 * @param world_size
 *
 * @return Tuple of (received buff, recv_sizes, recv_offset)
 */
// FIXME: wrap the low-level communicator
std::pair<IdArray, IdArray> Alltoall(
    int64_t rank, int64_t world_size, const IdArray& input, int64_t expand_size,
    const IdArray& send_offset, IdArray recv_offset, cudaStream_t stream);

}


#endif  // DGL_ALL2ALL_H
