#pragma once

#include <cstddef>
#include <cstdlib>
#include <cassert>

namespace Memory
{
    enum MallocType {Std, Managed};
    template<typename T>
    __host__ __device__ inline std::byte* align(std::size_t alignment, T const * ptr);
    __host__ __device__ inline std::byte* safeMalloc(std::size_t size, MallocType type);
    __host__ __device__ inline std::byte* safeStdMalloc(std::size_t size);
    __host__ inline std::byte* safeManagedMalloc(std::size_t size);
}
template<typename T>
__host__ __device__
std::byte* Memory::align(std::size_t alignment, T const * ptr)
{
    uintptr_t const intPtr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t const aligned = intPtr + (intPtr % alignment);
    return reinterpret_cast<std::byte*>(aligned);
}


__host__ __device__
std::byte* Memory::safeMalloc(std::size_t size, MallocType type)
{
#ifdef __CUDA_ARCH__
    switch(type)
    {
        case Std:
            return safeStdMalloc(size);
        default:
            assert(false);
            return nullptr;
    }
#else
    switch(type)
    {
        case Std:
            return safeStdMalloc(size);
        case Managed:
            return safeManagedMalloc(size);
        default:
            assert(false);
            return nullptr;
    }
#endif
}

__host__ __device__
std::byte* Memory::safeStdMalloc(std::size_t size)
{
    void* mem = malloc(size);
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
