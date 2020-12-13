#pragma once

#include <cstddef>
#include <cstdlib>
#include <cassert>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

namespace Memory
{
    enum MallocType {Std, Managed};
    __host__ __device__ inline std::byte* safeMalloc(std::size_t size, MallocType type = Std);
    __host__ __device__ inline std::byte* safeStdMalloc(std::size_t size);
    __host__            inline std::byte* safeManagedMalloc(std::size_t size);
    __host__ __device__ inline std::byte* align(std::size_t const alignment, std::byte const* ptr);

    __host__ __device__
    std::byte* safeMalloc(std::size_t size, MallocType type)
    {
#ifdef __CUDA_ARCH__
        switch(type)
        {
            case Std:
                return safeStdMalloc(size);
            default:
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
                return nullptr;
        }
#endif
    }

    __host__ __device__
    std::byte* safeStdMalloc(std::size_t size)
    {
        void* mem = malloc(size);
        assert(mem != nullptr);
        return static_cast<std::byte*>(mem);
    }

    __host__
    std::byte* safeManagedMalloc(std::size_t size)
    {
        void* mem;
        cudaError_t status = cudaMallocManaged(&mem, size);
        assert(status == cudaSuccess);
        assert(mem != nullptr);
        return static_cast<std::byte*>(mem);
    }

    __host__ __device__
    std::byte* align(std::size_t const alignment, std::byte const* ptr)
    {
        uintptr_t const intptr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t const aligned = intptr + (intptr % alignment);
        return reinterpret_cast<std::byte*>(aligned);
    }
}