cmake_minimum_required(VERSION 3.20)
project(TorchMetis)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
include_directories(${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(third_party/pybind11)
find_package(Python REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/metis_build/include)
link_directories(${CMAKE_SOURCE_DIR}/metis_build/lib)
include_directories(${CMAKE_SOURCE_DIR}/tbb_build/include)
link_directories(${CMAKE_SOURCE_DIR}/tbb_build/lib)

add_subdirectory(src)

message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message("CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
message("TORCH_INCLUD_DIRS ${TORCH_INCLUDE_DIRS}")
message("TORCH_LIBRARIES ${TORCH_LIBRARIES}")
message("TORCH_PYTHON_LIBRARY ${TORCH_PYTHON_LIBRARY}")