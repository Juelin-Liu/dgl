#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# build GKlib
pushd ${SCRIPT_DIR}/third_party/METIS/GKlib && make config prefix=${SCRIPT_DIR}/third_party/METIS/build openmp=set && make -j && make install && popd

# build METIS
pushd ${SCRIPT_DIR}/third_party/METIS && make config prefix=${SCRIPT_DIR}/third_party/METIS/build i64=1 && make -j && make install && popd

# build oneTBB
mkdir -p ${SCRIPT_DIR}/third_party/oneTBB/build
pushd ${SCRIPT_DIR}/third_party/oneTBB/build 
cmake -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DCMAKE_INSTALL_PREFIX=${SCRIPT_DIR}/third_party/oneTBB/build .. 
cmake --build . -j 
cmake --install . 
popd

# build NCCL
pushd third_party/nccl
make -j src.build NVCC_GENCODE="-arch=native"
popd
