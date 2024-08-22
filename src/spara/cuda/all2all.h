//
// Created by juelin on 2/14/24.
//

#ifndef DGL_ALL2ALL_H
#define DGL_ALL2ALL_H

#include <dgl/array.h>
#include <nccl.h>

namespace dgl::dev {

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
    int64_t rank, int64_t world_size, const IdArray& input,
    const IdArray& send_offset, IdArray recv_offset, cudaStream_t stream,
    ncclComm_t nccl_comm);
}


#endif  // DGL_ALL2ALL_H