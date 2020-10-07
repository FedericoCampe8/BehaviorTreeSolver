cmake_minimum_required(VERSION 3.16)

project(mdd-gpu)

# Add headers
set(HEADERS
    include/Extra/Containers.hh
    include/Extra/Containers/RestrainedArray.hh
    include/Extra/Containers/RestrainedVector.hh
    include/Extra/Extra.hh
    include/Extra/Algorithms.hh
    include/Extra/Utils/Chrono.hh
    include/Extra/Utils/Memory.hh
    include/Extra/Utils/Platform.hh
    include/Extra/Utils/Types.hh
    include/Extra/Utils.hh
    include/MDD/Edge.hh
    include/MDD/Layer.hh
    include/MDD/MDD.hh
    include/MDD/Node.hh
    include/Problem/AllDifferent.hh
    include/Problem/State.hh
    include/Problem/Variable.hh)

# Add sources
set(SOURCES
    src/Main.cx
    src/MDD/Edge.cx
    src/MDD/Layer.cx
    src/MDD/MDD.cx
    src/MDD/Node.cx
    src/Problem/AllDifferent.cx
    src/Problem/State.cx
    src/Problem/Variable.cx)


# Add headers
include_directories(include)

# Print information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform type: ${PLATFORM_TYPE}")

# Check TBB
find_package(TBB 2020.1 REQUIRED)

if(${PLATFORM_TYPE} STREQUAL "CPU")

    # C++
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    string(APPEND CMAKE_CXX_FLAGS " -Wall")

    # Check GCC minimum version
    set(GCC_MINIMUM_REQUIRED_VERSION 7)
    if (NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL GCC_MINIMUM_REQUIRED_VERSION)))
        message(FATAL_ERROR "Required GCC >= ${GCC_MINIMUM_REQUIRED_VERSION}")
    endif()

    # Compile source files as C++ files
    set_source_files_properties(${SOURCES} ${HEADERS} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${SOURCES} ${HEADERS} PROPERTIES LINKER_LANGUAGE CXX)
    set_source_files_properties(${SOURCES} ${HEADERS} PROPERTIES COMPILE_FLAGS " -x c++")

endif()


if(${PLATFORM_TYPE} STREQUAL "GPU")

    # CUDA
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    string(APPEND CMAKE_CUDA_FLAGS " -std c++17 -arch=sm_52")
    string(APPEND CMAKE_CUDA_FLAGS " -Xptxas -v")
    string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -G")
    string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -lineinfo")

    # Check CUDA and GCC version
    set(CUDA_REQUIRED_VERSION 11)
    if (NOT ((CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA") AND (CMAKE_CUDA_COMPILER_VERSION  MATCHES "^${CUDA_REQUIRED_VERSION}*")))
        message(FATAL_ERROR "Required CUDA ${CUDA_REQUIRED_VERSION}")
    endif()
    set(GCC_REQUIRED_VERSION 9)
    if (NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND (CMAKE_CXX_COMPILER_VERSION MATCHES "^${GCC_REQUIRED_VERSION}*")))
        message(FATAL_ERROR "Required GCC ${GCC_REQUIRED_VERSION}")
    endif()

    # Compile source files as CUDA files
    set_source_files_properties(${SOURCES} ${HEADERS} PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${SOURCES} ${HEADERS} PROPERTIES LINKER_LANGUAGE CUDA)
    set_source_files_properties(${SOURCES} ${HEADERS} PROPERTIES COMPILE_FLAGS " -x cu")

    # Define GPU macro
    add_definitions(-DGPU)
endif()

# Add target executable
add_executable(mdd-gpu ${SOURCES} ${HEADERS})

# Link TBB
target_link_libraries(mdd-gpu ${TBB_IMPORTED_TARGETS})