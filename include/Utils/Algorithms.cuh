#pragma once

#include <Utils/CUDA.cuh>
#include <Utils/TypeAlias.h>

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
    __device__ inline void oddEvenSort(T* array, unsigned int size);
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
__device__
void Algorithms::oddEvenSort(T* array, unsigned int size)
{
    u32 const threads = blockDim.x;
    u32 const pairs = size / 2 + (size % 2 != 0);
    u32 const pairsPerThread = CUDA::getElementsPerThread(threads, pairs);
    u32 const elementsPerThread = 2 * pairsPerThread;

    for (unsigned int iterationIdx = 0; iterationIdx < pairs + 1; iterationIdx += 1)
    {
        __syncthreads();
        u32 beginIdx = threadIdx.x * elementsPerThread;
        u32 endIdx = Algorithms::min(beginIdx + elementsPerThread, size - 1);
        for(u32 i = beginIdx; i < endIdx; i += 2)
        {
            if (not(array[i] < array[i+1]))
            {
                T::swap(array[i], array[i+1]);
            }
        }
        __syncthreads();
        beginIdx = threadIdx.x * elementsPerThread + 1;
        endIdx = Algorithms::min(beginIdx + elementsPerThread + 1, size - 1);
        for(u32 i = beginIdx; i < endIdx; i += 2)
        {
            if (not(array[i] < array[i+1]))
            {
                T::swap(array[i], array[i+1]);
            }
        }
    }
}