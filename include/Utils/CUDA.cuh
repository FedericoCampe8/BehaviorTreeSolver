#pragma once

#include <Utils/TypeAlias.h>

#ifdef __CUDA_ARCH__
    #define CUDA_THREADS_BARRIER __syncthreads();
    #define CUDA_ONLY_FIRST_THREAD if (threadIdx.x == 0)
#else
    #define CUDA_THREADS_BARRIER
    #define CUDA_ONLY_FIRST_THREAD
#endif

namespace CUDA
{
    __host__ __device__ inline u32 getElementsPerThread(u32 const threads, u32 const elements);
}

__host__ __device__
u32 CUDA::getElementsPerThread(u32 const threads, u32 const elements)
{
    if (threads >= elements)
    {
        return 1;
    }
    else
    {
        return (elements / threads) + 1;
    }
}