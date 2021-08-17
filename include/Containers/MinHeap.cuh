#pragma once

#include "Containers/Vector.cuh"
#include <thrust/swap.h>

template<typename T>
class MinHeap
{
    // Members
    private:
    Vector<T> vector;

    // Functions
    public:
    __host__ __device__ MinHeap(u32 capacity, Memory::MallocType mallocType);
    __host__ __device__ void popFront();
    __host__ __device__ inline T* front() const;
    __host__ __device__ inline T* at(u32 index) const;
    __host__ __device__ inline u32 getSize() const;
    __host__ __device__ void insert(T const * t);
    __host__ __device__ inline bool isEmpty() const;
    __host__ __device__ inline bool isFull() const;
    private:
    __host__ __device__ void heapify(u32 index);
    __host__ __device__ inline u32 left(u32 index);
    __host__ __device__ inline u32 parent(u32 index);
    __host__ __device__ inline u32 right(u32 index);
};

template<typename T>
__host__ __device__
MinHeap<T>::MinHeap(u32 capacity, Memory::MallocType mallocType) :
    vector(capacity, mallocType)
{}

template<typename T>
__host__ __device__
void MinHeap<T>::popFront()
{
    thrust::swap(*vector.front(), *vector.back());
    vector.popBack();
    heapify(0);
}

template<typename T>
__host__ __device__
T* MinHeap<T>::front() const
{
    return vector.front();
}

template<typename T>
__host__ __device__
void MinHeap<T>::insert(T const * t)
{
    vector.pushBack(t);
    u32 i = vector.getSize() - 1;
    u32 p = parent(i);
    while (i > 0 and *vector[i] < *vector[p])
    {
        thrust::swap(*vector[p], *vector[i]);
        i = p;
        p = parent(i);
    }
}

template<typename T>
__host__ __device__
bool MinHeap<T>::isEmpty() const
{
    return vector.isEmpty();
}

template<typename T>
__host__ __device__
bool MinHeap<T>::isFull() const
{
    return vector.isFull();
}

template<typename T>
__host__ __device__
void MinHeap<T>::heapify(u32 index)
{
    u32 const l = left(index);
    u32 const r = right(index);
    u32 smallest = index;
    if (l < vector.getSize() and (*vector[l] < *vector[index]))
    {
        smallest = l;
    }
    if (r < vector.getSize() and (*vector[r] < *vector[smallest]))
    {
        smallest = r;
    }
    if (smallest != index)
    {
        thrust::swap(*vector[index], *vector[smallest]);
        heapify(smallest);
    }
}

template<typename T>
__host__ __device__
u32 MinHeap<T>::left(u32 index)
{
    return 2 * index + 1;
}

template<typename T>
__host__ __device__
u32 MinHeap<T>::parent(u32 index)
{
    return (index - 1) / 2;
}

template<typename T>
__host__ __device__
u32 MinHeap<T>::right(u32 index)
{
    return 2 * index + 2;
}

template<typename T>
__host__ __device__
T* MinHeap<T>::at(u32 index) const
{
    return vector.at(index);
}
template<typename T>
__host__ __device__
u32 MinHeap<T>::getSize() const
{
    return vector.getSize();
}
