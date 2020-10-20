#pragma once

#include <cstddef>
#include <cstdlib>
#include <cassert>

namespace Memory
{
    const uint16_t DefaultAlignment = 64;

    __host__ __device__ inline void * alignedMalloc(std::size_t size, uint16_t alignment = DefaultAlignment);
}

__host__ __device__
void* Memory::alignedMalloc(std::size_t size, uint16_t alignment)
{
    void * mem =
#ifdef __CUDA_ARCH__
        __nv_aligned_device_malloc(size, alignment);
#else
        aligned_alloc(alignment, size);
#endif
    assert(mem != nullptr);
    return mem;
}