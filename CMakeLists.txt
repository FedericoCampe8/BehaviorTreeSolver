cmake_minimum_required(VERSION 3.18)

project(mdd-gpu LANGUAGES CXX)

# CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -Wall")
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -G")
string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -lineinfo")

# Check CUDA and GCC version
set(CUDA_REQUIRED_VERSION 11)
set(GCC_REQUIRED_VERSION 9)
if (NOT ((CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA") AND (CMAKE_CUDA_COMPILER_VERSION MATCHES "^${CUDA_REQUIRED_VERSION}*")))
    message(FATAL_ERROR "Required CUDA ${CUDA_REQUIRED_VERSION}")
endif ()
if (NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND (CMAKE_CXX_COMPILER_VERSION MATCHES "^${GCC_REQUIRED_VERSION}*")))
    message(FATAL_ERROR "Required GCC ${GCC_REQUIRED_VERSION}")
endif ()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Add sources
set(SOURCES
    src/Main.cu
    src/OP/Variable.cuh
    src/DP/CTWPModel.cuh
    src/DP/SOPModel.cuh
    src/DD/MDD.cuh
    src/OP/CTWProblem.cuh
    src/OP/SOProblem.cuh
    src/OP/Problem.cuh
    src/DP/State.cuh
    src/LNS/OffloadBuffer.cuh
    src/LNS/SearchManagerCPU.cuh
    src/LNS/SearchManagerGPU.cuh
    src/LNS/Neighbourhood.cuh
    src/LNS/SyncState.cuh
    src/OP/Context.h
    src/DP/Context.h
    src/DD/StateMetadata.cuh
    src/LNS/StatesPriorityQueue.cuh
    src/DD/Context.h
    include/Utils/Algorithms.cuh
    include/Containers/Pair.cuh
    src/Options.h
    src/DP/CTWPState.cuh
    src/DP/SOPState.cuh
    src/OP/OSSProblem.cuh
    src/DP/OSSPState.cuh
    src/DP/OSSPModel.cuh)


# Add include path
include_directories(src)
include_directories(include)

# Add target executable
add_executable(mdd-gpu ${SOURCES})