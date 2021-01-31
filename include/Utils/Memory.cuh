#pragma once

#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <Utils/TypeAlias.h>

namespace Memory
{
    enum MallocType {Managed, Std};
    u32 const DefaultAlignment = 8;
    template<typename T>
    __host__ __device__ inline T* align(std::byte const * ptr);
    __host__ __device__ inline std::byte* align(std::byte const * ptr, u32 alignment = DefaultAlignment);
    __host__ __device__ std::byte* safeMalloc(unsigned int size, MallocType type);
    __host__ __device__ std::byte* safeStdMalloc(unsigned int size);
    std::byte* safeManagedMalloc(unsigned int size);
}

template<typename T>
__host__ __device__
T* Memory::align(std::byte const * ptr)
{
    uintptr_t const address = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t const aligned = address + (address % sizeof(T));
    return reinterpret_cast<T*>(aligned);
}

__host__ __device__
std::byte* Memory::align(std::byte const * ptr, u32 alignment)
{
    uintptr_t const address = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t const aligned = address + (address % alignment);
    return reinterpret_cast<std::byte*>(aligned);
}

__host__ __device__
std::byte* Memory::safeMalloc(unsigned int size, MallocType type)
{
    switch (type)
    {
        case Std:
            return safeStdMalloc(size);
#ifndef __CUDA_ARCH__
        case Managed:
            return safeManagedMalloc(size);
#endif
        default:
            assert(false);
            return nullptr;
    }
}

__host__ __device__
std::byte* Memory::safeStdMalloc(unsigned int size)
{
    void* memory = malloc(size);
    assert(memory != nullptr);
    return static_cast<std::byte*>(memory);
}

std::byte* Memory::safeManagedMalloc(unsigned int size)
{
    void* memory;
    cudaError_t status = cudaMallocManaged(& memory, size);
    assert(status == cudaSuccess);
    assert(memory != nullptr);
    return static_cast<std::byte*>(memory);
}