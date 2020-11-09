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
    template<typename T>
    __host__ __device__ inline std::byte* endOf(T const * t);


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

template<typename T>
__host__ __device__
std::byte* Memory::endOf(T const * t)
{
    return reinterpret_cast<std::byte*>(t) + sizeof(T);
}