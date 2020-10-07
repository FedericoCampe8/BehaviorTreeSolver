#pragma once

#include <cstdlib>

#include <Extra/Utils/Platform.hh>

namespace Extra::Utils::Memory
{
    __host__ __device__
    inline void malloc(void ** ptr, std::size_t size)
    {
#if defined(__CUDA_ARCH__) || !defined(GPU)
        *ptr = std::malloc(size);
#else
        cudaError_t status = cudaMallocManaged(ptr, size);
        Extra::Utils::Platform::assert(status == cudaSuccess);
#endif
        Extra::Utils::Platform::assert(*ptr != nullptr);
    }

    __host__ __device__
    inline void * malloc(std::size_t size)
    {
        void * ptr;
        malloc(&ptr, size);
        return ptr;
    }

    __host__ __device__
    inline void free(void * const ptr)
    {
#if defined(__CUDA_ARCH__) || !defined(GPU)
        std::free(ptr);
#else
        cudaError_t status = cudaFree(ptr);
        Extra::Utils::Platform::assert(status == cudaSuccess);
#endif
    }
}