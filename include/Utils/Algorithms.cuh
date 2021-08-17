#pragma once

#include <thrust/sort.h>
#include <Utils/TypeAlias.h>

namespace Algorithms
{
    template<typename T>
    __host__ __device__ inline void sort(T* array, u32 size);
    __host__ __device__ inline u32 nextPower2(u32 n);
}


template<typename T>
__host__ __device__
void Algorithms::sort(T* array, u32 size)
{
#ifdef __CUDA_ARCH__

    for (u32 k = 2; k <= size; k *= 2)
    {
        for (u32 j = k / 2; j > 0; j /= 2)
        {
            u32 const i = threadIdx.x;
            u32 const l = i ^ j;
            if (l > i)
            {
                if ((i & k) == 0)
                {
                    if (array[l] < array[i])
                    {
                        thrust::swap(array[i], array[l]);
                    }
                }
                else
                {
                    if (array[i] < array[l])
                    {
                        thrust::swap(array[i], array[l]);
                    }
                }

            }
            __syncthreads();
        }
    }
    __syncthreads();
#else
    thrust::sort(thrust::host, array, array + size);
#endif
}

__host__ __device__
u32 Algorithms::nextPower2(u32 n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}