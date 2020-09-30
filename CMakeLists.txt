cmake_minimum_required(VERSION 3.16)

project(mdd-gpu)

# Set build types
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Available builds")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Debug)

# Set platform types
set(PLATFORM_TYPE "CPU" CACHE STRING "Available platforms")
set_property(CACHE PLATFORM_TYPE PROPERTY STRINGS CPU GPU)

# Set GCC, TBB and CUDA minimum version
set(GCC_MINIMUM_REQUIRED_VERSION 9.3)
set(CUDA_MINIMUM_REQUIRED_VERSION 10.1)

# Add headers
set(HEADERS
    include/CustomTemplateLibrary/CTL.hh
        include/CustomTemplateLibrary/StaticVector.hh
    include/CustomTemplateLibrary/Types.hh
        include/DP/AllDifferent.hh
    include/MDD/MDD.hh
    include/MDD/Edge.hh
    include/MDD/Layer.hh
    include/MDD/Node.hh
    include/Problem/Variable.hh
        include/DP/State.hh)

#Add sources
set(SOURCES
        src/Main.cc
        src/MDD/MDD.cc
        src/MDD/Edge.cc
        src/MDD/Layer.cc
        src/MDD/Node.cc
        src/Problem/Variable.cc
        src/DP/AllDifferent.cc
        src/DP/State.cc
        )

#Add include directory
include_directories("include")

# Print information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform type: ${PLATFORM_TYPE}")

#Set C++ compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g0 -O3")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0")

# CPU
if(${PLATFORM_TYPE} STREQUAL "CPU")

    # Check GCC version
    if ((NOT CMAKE_COMPILER_IS_GNUCC) OR (CMAKE_CXX_COMPILER_VERSION VERSION_LESS GCC_MINIMUM_REQUIRED_VERSION))
        message(FATAL_ERROR "Required GCC >= ${GCC_MINIMUM_REQUIRED_VERSION}")
    endif()

    # Check Intel Threading Building Blocks presence
    find_package(TBB 2020.1 REQUIRED)

    # Add target executable
    add_executable(mdd-gpu ${SOURCES} ${HEADERS})

    # Link TBB library
    target_link_libraries(mdd-gpu ${TBB_IMPORTED_TARGETS})

#GPU
else()

    # Check CUDA presence
    find_package(CUDA ${CUDA_MINIMUM_REQUIRED_VERSION} REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})

    # Define the macro GPU
    add_definitions(-DGPU)

    # Set CUDA compilation flags
    cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
    list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-G")
        add_definitions(-DTHRUST_DEBUG)
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-lineinfo")
    endif()

    set(CUDA_SEPARABLE_COMPILATION ON)

    # Add target executable
    cuda_add_executable(mdd-gpu ${SOURCES} ${HEADERS})

    # Link CUDA Runtime library
    target_link_libraries(mdd-gpu ${CUDA_cudadevrt_LIBRARY})

endif()
