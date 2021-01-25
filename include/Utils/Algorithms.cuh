#pragma once

namespace Algorithms
{
    template<typename T>
    __device__ inline void oddEvenSort(T* array, unsigned int size);
}

template<typename T>
__device__
void Algorithms::oddEvenSort(T* array, unsigned int size)
{
    for (unsigned int iterationIdx = 0; iterationIdx < (size / 2) + 1; iterationIdx += 1)
    {
        __syncthreads();
        unsigned int const i = threadIdx.x;
        if (i < size / 2)
        {
            unsigned int const idx0 = 2 * i;
            unsigned int const idx1 = (2 * i) + 1;
            if (not(array[idx0] < array[idx1]))
            {
                T::swap(array[idx0], array[idx1]);
            }
        }
        __syncthreads();
        if (i >= size / 2 and i <= size - 2)
        {
            unsigned int const j = i - (size / 2);
            unsigned int const idx0 = 1 + (2 * j);
            unsigned int const idx1 = 1 + (2 * j) + 1;
            if (not(array[idx0] < array[idx1]))
            {
                T::swap(array[idx0], array[idx1]);
            }
        }
    }
}