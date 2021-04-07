#pragma once

namespace Algorithms
{
    template<typename T>
    __host__ __device__ T& min(T& a, T& b);

    template<typename T>
    __host__ __device__ T& min(T const & a, T const & b);

    template<typename T>
    __host__ __device__ T& max(T& a, T& b);

    template<typename T>
    __host__ __device__ T& max(T const & a, T const& b);

    template<typename T>
    __device__ inline void oddEvenSort(T* array, unsigned int size);

    template<typename T>
    __host__ __device__ void orderedInsertion(T* array, unsigned int size, T const * value);
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

template<typename T>
__host__ __device__
void Algorithms::orderedInsertion(T* array, unsigned int size, T const * value)
{
    for (int index = size - 1; index >= 0; index -= 1)
    {
        if (not (array[index] < *value))
        {
            T::swap(array[index], array[index+1]);
        }
        else
        {
            array[index] = *value;
            break;
        }
    }
}