#pragma once

#include <thrust/swap.h>
#include <thrust/distance.h>

template<typename T>
class StaticMaxHeap
{
    public:
        using Comparator = bool(*)(T const & t0, T const & t1);

    public:
        Comparator const comparator;
        StaticVector<T> vector;

    public:
        __host__ __device__ StaticMaxHeap(unsigned int capacity, Comparator comparator, Memory::MallocType mallocType);
        __host__ __device__ void eraseIndex(unsigned int index);
        __host__ __device__ void insertBack();

    private:
        __host__ __device__ inline unsigned int parent(unsigned int index);
        __host__ __device__ inline unsigned int left(unsigned int index);
        __host__ __device__ inline unsigned int right(unsigned int index);
        __host__ __device__ void heapify(unsigned int index);
};

template<typename T>
__host__ __device__
StaticMaxHeap<T>::StaticMaxHeap(unsigned int capacity, Comparator comparator, Memory::MallocType mallocType) :
    comparator(comparator),
    vector(capacity, mallocType)
{}

template<typename T>
__host__ __device__
void StaticMaxHeap<T>::erase(unsigned int index)
{
    while(index > 0)
    {
        unsigned int p = parent(index);
        thrust::swap(vector[index], vector[p]);
        index = p;
    }

    vector[0] = std::move(vector.back());
    vector.popBack();
    heapify(0);
}

template<typename T>
__host__ __device__
void StaticMaxHeap<T>::insertBack()
{
    unsigned int i = vector.getSize() - 1;
    unsigned int p = parent(index);
    while(index > 0 and (not cmp(vector[p], vector[i])))
    {
        thrust::swap(vector[i], vector[p]);
        i = p;
        p = parent(i);
    }
}

template<typename T>
__host__ __device__
unsigned int StaticMaxHeap<T>::parent(unsigned int index)
{
    return (i - 1) / 2;
}

template<typename T>
__host__ __device__
unsigned int StaticMaxHeap<T>::left(unsigned int index)
{
    return 2 * i + 1;
}

template<typename T>
__host__ __device__
unsigned int StaticMaxHeap<T>::right(unsigned int index)
{
    return 2 * i + 2;
}

template<typename T>
__host__ __device__
void StaticMaxHeap<T>::heapify(unsigned int index)
{
    unsigned int l = left(index);
    unsigned int r = right(index);
    unsigned int largest = index;

    if (l < vector.getSize() && cmp(vector[l],vector[i]))
    {
        largest = l;
    }

    if (r < vector.getSize() && cmp(vector[r],vector[largest]))
    {
        largest = r;
    }

    if (largest != index)
    {
        thrust::swap(vector[index], vector[largest]);
        heapify(largest);
    }
}