#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# build GKlib
rm -rf ${SCRIPT_DIR}/third_party/build
pushd ${SCRIPT_DIR}/third_party/METIS/GKlib
make config prefix=${SCRIPT_DIR}/third_party/build openmp=set
make -j 
make install 
popd

# build METIS
pushd ${SCRIPT_DIR}/third_party/METIS
make config prefix=${SCRIPT_DIR}/third_party/build i64=1
make -j
make install
popd

# build oneTBB
rm -rf ${SCRIPT_DIR}/third_party/oneTBB/build
mkdir -p ${SCRIPT_DIR}/third_party/oneTBB/build
pushd ${SCRIPT_DIR}/third_party/oneTBB/build 
cmake -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DCMAKE_INSTALL_PREFIX=${SCRIPT_DIR}/third_party/build .. 
cmake --build . -j 
cmake --install . 
popd

# build NCCL
pushd ${SCRIPT_DIR}/third_party/nccl
rm -rf ${SCRIPT_DIR}/third_party/nccl/build
make -j src.build NVCC_GENCODE="-arch=native" BUILDDIR=${SCRIPT_DIR}/third_party/build 
popd
