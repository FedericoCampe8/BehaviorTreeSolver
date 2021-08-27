#pragma once

#include <cassert>
#include <algorithm>
#include <random>
#include <Utils/TypeAlias.h>
#include <Containers/Pair.cuh>

namespace Algorithms
{
    template<typename T>
    __device__ inline void sort(T* begin, T* end);
    __host__ __device__ Pair<u32> getBeginEndIndices(u32 index, u32 threads, u32 elements);
}


template<typename T>
__device__
void Algorithms::sort(T* begin, T* end)
{
    T* array = begin;
    u32 size = end - begin;
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
}

__host__ __device__
Pair<u32> Algorithms::getBeginEndIndices(u32 index, u32 threads, u32 elements)
{
    u32 const elementsPerThread = (elements + threads - 1) / threads;
    u32 const beginIdx = elementsPerThread * index;
    u32 const endIdx = min(beginIdx + elementsPerThread, elements);
    return Pair<u32>(beginIdx, endIdx);
}
