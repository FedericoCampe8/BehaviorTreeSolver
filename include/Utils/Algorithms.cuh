#pragma once

#include <Utils/CUDA.cuh>
#include <Utils/TypeAlias.h>
#include <thrust/swap.h>

namespace Algorithms
{
    template<typename T>
    __host__ __device__ inline T& min(T& a, T& b);

    template<typename T>
    __host__ __device__ inline T& min(T const & a, T const & b);

    template<typename T>
    __host__ __device__ inline T& max(T& a, T& b);

    template<typename T>
    __host__ __device__ inline T& max(T const & a, T const& b);

    template<typename T>
    __host__ __device__ inline void oddEvenSort(T* array, unsigned int size);
}

template<typename T>
__host__ __device__
T& Algorithms::min(T const & a, T const & b)
{
    return min(const_cast<T&>(a), const_cast<T&>(b));
}

template<typename T>
__host__ __device__
T& Algorithms::min(T& a, T& b)
{
    return a < b ? a : b;
}

template<typename T>
__host__ __device__
T& Algorithms::max(T const & a, T const & b)
{
    return max(const_cast<T&>(a), const_cast<T&>(b));
}

template<typename T>
__host__ __device__
T& Algorithms::max(T& a, T& b)
{
    return a > b ? a : b;
}

template<typename T>
__host__ __device__
void Algorithms::oddEvenSort(T* array, unsigned int size)
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
#else
    bool sorted;
    u32 const pairs = size / 2;

    do
    {
        sorted = true;
        for(u32 pairIdx = 0; pairIdx < pairs; pairIdx += 1)
        {
            u32 const idx0 = pairIdx * 2;
            u32 const idx1 = pairIdx * 2 + 1;
            if(array[idx1] < array[idx0])
            {
                T::swap(array[idx1],array[idx0]);
                sorted = false;
            }
        }

        for(u32 pairIdx = 0; pairIdx < pairs; pairIdx += 1)
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

    }
    while (not sorted);

#endif
}