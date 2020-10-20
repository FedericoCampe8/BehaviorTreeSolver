cmake_minimum_required(VERSION 3.16)

project(mdd-gpu)

# CUDA
enable_language(CUDA)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 52)
else()
    string(APPEND CMAKE_CUDA_FLAGS " --std=c++17")
    string(APPEND CMAKE_CUDA_FLAGS " --gpu-architecture=sm_52")
endif()
string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
string(APPEND CMAKE_CUDA_FLAGS " --compiler-options=-Wall")
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -G")
string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -lineinfo")

# Check CUDA and GCC version
set(CUDA_REQUIRED_VERSION 11)
set(GCC_REQUIRED_VERSION 9)
if (NOT ((CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA") AND (CMAKE_CUDA_COMPILER_VERSION MATCHES "^${CUDA_REQUIRED_VERSION}*")))
    message(FATAL_ERROR "Required CUDA ${CUDA_REQUIRED_VERSION}")
endif()
if (NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND (CMAKE_CXX_COMPILER_VERSION MATCHES "^${GCC_REQUIRED_VERSION}*")))
    message(FATAL_ERROR "Required GCC ${GCC_REQUIRED_VERSION}")
endif()

# Add headers
set(HEADERS
    include/Extra/Containers/StaticVector.cuh
    include/Extra/Utils/Chrono.cuh
    include/MDD/Edge.cuh
    include/MDD/MDD.cuh
    include/Problem/AllDifferent.cuh
    include/Problem/DPModel.cuh
    include/Problem/Variable.cuh
    include/Extra/Utils/CUDA.cuh
    include/Extra/Utils/Algorithms.cuh
    include/Extra/Utils/Memory.cuh)

# Add sources
set(SOURCES
    src/Main.cu
    src/MDD/Edge.cu
    src/Problem/AllDifferent.cu
        src/Problem/Variable.cu)

# Add include path
include_directories(include)

# Add target executable
add_executable(mdd-gpu ${SOURCES} ${HEADERS})