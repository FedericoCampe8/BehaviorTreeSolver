#pragma once

#include <thrust/sort.h>
#include <Utils/TypeAlias.h>

namespace Algorithms
{
    template<typename T>
    __host__ __device__ void sort(T* array, u32 size);
}


template<typename T>
__host__ __device__
void Algorithms::sort(T* array, u32 size)
{
#ifdef __CUDA_ARCH__
    __shared__ bool sorted;
    u32 const threads = blockDim.x;
    u32 const pairs = size / 2;
    do
    {
        __syncthreads();
        if(threadIdx.x == 0)
        {
            sorted = true;
        }
        __syncthreads();
        for(u32 pairIdx = threadIdx.x; pairIdx < pairs; pairIdx += threads)
        {
            u32 const idx0 = pairIdx * 2;
            u32 const idx1 = pairIdx * 2 + 1;
            if(array[idx1] < array[idx0])
            {
                T::swap(array[idx1],array[idx0]);
                sorted = false;
            }
        }
        __syncthreads();

        for(u32 pairIdx = threadIdx.x; pairIdx < pairs; pairIdx += threads)
        {
            u32 const idx0 = 1 + pairIdx * 2;
            u32 const idx1 = 1 + pairIdx * 2 + 1;
            if(idx1 < size)
            {
                if(array[idx1] < array[idx0])
                {
                    T::swap(array[idx1],array[idx0]);
                    sorted = false;
                }
            }
        }
        __syncthreads();
    }
    while (not sorted);
    __syncthreads();
#else
    thrust::sort(thrust::host, array, array + size);
#endif

}
