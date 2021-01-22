#pragma once


namespace Algorithms
{
    template<typename T>
    __host__ __device__ void quickSort(T* array, unsigned int size);

    template<typename T>
    __host__ __device__ void quickSort(T* array, int first, int last);

    template<typename T>
    __host__ __device__ int partition(T* array, int first, int last);

    template<typename T>
    __device__ inline void oddEvenSort(T* array, unsigned int size);
}

template<typename T>
void Algorithms::quickSort(T* array, unsigned int size)
{
    quickSort(array, 0, size - 1);
}

template<typename T>
void Algorithms::quickSort(T* array, int first, int last)
{
    if (first < last)
    {
        int p = partition(array, first, last);
        quickSort(array, first, p - 1);
        quickSort(array, p + 1, last);
    }
}

template<typename T>
int Algorithms::partition(T* array, int first, int last)
{
    T const & pivot = array[last];
    int i = (first - 1);

    for (int j = first; j <= last - 1; j += 1)
    {
        if (array[j] < pivot)
        {
            i += 1; // increment index of smaller element
            T::swap(array[i], array[j]);
        }
    }
    T::swap(array[i + 1], array[last]);
    return (i + 1);
}

template<typename T>
__device__
void Algorithms::oddEvenSort(T* array, unsigned int size)
{
    for(unsigned int round = 0; round < size; round += 1)
    {
        __syncthreads();
        unsigned int const i = threadIdx.x;
        if (i % 2 == 0 and i <= size - 2)
        {
            if (not (array[i] < array[i + 1]))
            {
                T::swap(array[i], array[i + 1]);
            }
        }
        __syncthreads();
        if (i % 2 == 1 and i <= size - 2)
        {
            if (not (array[i] < array[i + 1]))
            {
                T::swap(array[i], array[i + 1]);
            }
        }

    }
}