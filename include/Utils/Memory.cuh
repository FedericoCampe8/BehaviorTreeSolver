#pragma once

#include <cstddef>
#include <cstdlib>
#include <cassert>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

namespace Memory
{
    __host__ __device__ inline std::byte* safeMalloc(std::size_t size);
    __host__ inline std::byte* safeManagedMalloc(std::size_t size);
    __host__ __device__ inline std::byte*  align(size_t const  alignment, std::byte const * ptr);

}

__host__ __device__
std::byte* Memory::safeMalloc(std::size_t size)
{
    void* mem = std::malloc(size);
    assert(mem != nullptr);
    return static_cast<std::byte*>(mem);
}

__host__
std::byte* Memory::safeManagedMalloc(std::size_t size)
{
    void* mem;
    cudaError_t status = cudaMallocManaged(&mem, size);
    assert(status == cudaSuccess);
    assert(mem != nullptr);
    return static_cast<std::byte*>(mem);
}

__host__ __device__
std::byte* Memory::align(size_t const alignment, std::byte const * ptr)
{
    uintptr_t const intptr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t const aligned = intptr + (intptr  % alignment);
    return reinterpret_cast<std::byte*>(aligned);
}