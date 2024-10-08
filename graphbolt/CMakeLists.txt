cmake_minimum_required(VERSION 3.18)
project(graphbolt C CXX)
set (CMAKE_CXX_STANDARD 17)

if(USE_CUDA)
  message(STATUS "Build graphbolt with CUDA support")
  enable_language(CUDA)
  add_definitions(-DGRAPHBOLT_USE_CUDA)
endif()

# Find PyTorch cmake files and PyTorch versions with the python interpreter
# $PYTHON_INTERP ("python3" or "python" if empty)
if(NOT PYTHON_INTERP)
  find_program(PYTHON_INTERP NAMES python3 python)
endif()

message(STATUS "Using Python interpreter: ${PYTHON_INTERP}")

file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/find_cmake.py FIND_CMAKE_PY)
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_CMAKE_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "find_cmake.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)

message(STATUS "Configuring for PyTorch ${TORCH_VER}")
string(REPLACE "." ";" TORCH_VERSION_LIST ${TORCH_VER})

set(Torch_DIR "${TORCH_PREFIX}/Torch")
message(STATUS "Setting directory to ${Torch_DIR}")

find_package(Torch REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TORCH_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -ggdb")

set(LIB_GRAPHBOLT_NAME "graphbolt_pytorch_${TORCH_VER}")

set(BOLT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(BOLT_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
file(GLOB BOLT_HEADERS ${BOLT_INCLUDE})
file(GLOB BOLT_SRC ${BOLT_DIR}/*.cc)
if(USE_CUDA)
  file(GLOB BOLT_CUDA_SRC
    ${BOLT_DIR}/cuda/*.cu
  )
  list(APPEND BOLT_SRC ${BOLT_CUDA_SRC})
endif()

add_library(${LIB_GRAPHBOLT_NAME} SHARED ${BOLT_SRC} ${BOLT_HEADERS})
target_include_directories(${LIB_GRAPHBOLT_NAME} PRIVATE ${BOLT_DIR}
                           ${BOLT_HEADERS}
                           "../third_party/dmlc-core/include"
                           "../third_party/pcg/include")
target_link_libraries(${LIB_GRAPHBOLT_NAME} "${TORCH_LIBRARIES}")

if(USE_CUDA)
  set_target_properties(${LIB_GRAPHBOLT_NAME} PROPERTIES CUDA_STANDARD 17)
  message(STATUS "Use external CUB/Thrust library for a consistent API and performance for graphbolt.")
  target_compile_definitions(${LIB_GRAPHBOLT_NAME} PRIVATE CUB_WRAPPED_NAMESPACE=graphbolt)
  target_compile_definitions(${LIB_GRAPHBOLT_NAME} PRIVATE THRUST_NS_QUALIFIER=thrust)
  target_include_directories(${LIB_GRAPHBOLT_NAME} PRIVATE
                             "../third_party/thrust"
                             "../third_party/thrust/dependencies/cub"
                             "../third_party/thrust/dependencies/libcudacxx/include")
endif()

# The Torch CMake configuration only sets up the path for the MKL library when
# using the conda distribution. The following is a workaround to address this
# when using a standalone installation of MKL.
if(DEFINED MKL_LIBRARIES)
  target_link_directories(${LIB_GRAPHBOLT_NAME} PRIVATE
                          ${MKL_ROOT}/lib/${MKL_ARCH})
endif()

