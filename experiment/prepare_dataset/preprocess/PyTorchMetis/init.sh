#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# echo "Script directory: $SCRIPT_DIR"
# build GKlib
cd ${SCRIPT_DIR}/third_party/GKlib && make config prefix=${SCRIPT_DIR}/metis_build openmp=set && make -j && make install
# build METIS
cd ${SCRIPT_DIR}/third_party/METIS && make config prefix=${SCRIPT_DIR}/metis_build i64=1 && make -j && make install
# build oneTBB
mkdir -p ${SCRIPT_DIR}/third_party/oneTBB/build
mkdir -p ${SCRIPT_DIR}/tbb_build

cd ${SCRIPT_DIR}/third_party/oneTBB/build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$SCRIPT_DIR/tbb_build -DTBB_TEST=OFF ..
cd ${SCRIPT_DIR}/third_party/oneTBB/build && cmake --build . -j
cd ${SCRIPT_DIR}/third_party/oneTBB/build && cmake --install .