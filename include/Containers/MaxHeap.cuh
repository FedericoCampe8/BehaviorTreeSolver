#pragma once

#include <thrust/swap.h>
#include <thrust/distance.h>

template<typename T>
class MaxHeap
{
    public:
        using Comparator = bool(*)(T const & t0, T const & t1);

    private:
        Vector<T> vector;
        Comparator const cmp;

    public:
        __host__ __device__ MaxHeap(unsigned int capacity, Comparator cmp, Memory::MallocType mallocType);
        __host__ __device__ void erase(unsigned int index);
        __host__ __device__ inline T& front() const;
        __host__ __device__ inline unsigned int getSize();
        __host__ __device__ void insertBack();
        __host__ __device__ void pushBack(T const & t);

    private:
        __host__ __device__ inline unsigned int parent(unsigned int index);
        __host__ __device__ inline unsigned int left(unsigned int index);
        __host__ __device__ inline unsigned int right(unsigned int index);
        __host__ __device__ void heapify(unsigned int index);
};

template<typename T>
__host__ __device__
MaxHeap<T>::MaxHeap(unsigned int capacity, Comparator cmp, Memory::MallocType mallocType) :
    cmp(cmp),
    vector(capacity, mallocType)
{}

template<typename T>
__host__ __device__
void MaxHeap<T>::erase(unsigned int index)
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
T& MaxHeap<T>::front() const
{
    return vector.front();
}


template<typename T>
__host__ __device__
unsigned int MaxHeap<T>::getSize()
{
    return vector.getSize();
}

template<typename T>
__host__ __device__
void MaxHeap<T>::insertBack()
{
    unsigned int i = vector.getSize() - 1;
    unsigned int p = parent(i);
    while(i > 0 and (not cmp(vector[p], vector[i])))
    {
        thrust::swap(vector[i], vector[p]);
        i = p;
        p = parent(i);
    }
}


template<typename T>
__host__ __device__
void MaxHeap<T>::pushBack(const T& t)
{
    vector.pushBack(t);
}

template<typename T>
__host__ __device__
unsigned int MaxHeap<T>::parent(unsigned int index)
{
    return (index - 1) / 2;
}

template<typename T>
__host__ __device__
unsigned int MaxHeap<T>::left(unsigned int index)
{
    return 2 * index + 1;
}

template<typename T>
__host__ __device__
unsigned int MaxHeap<T>::right(unsigned int index)
{
    return 2 * index + 2;
}

template<typename T>
__host__ __device__
void MaxHeap<T>::heapify(unsigned int index)
{
    unsigned int l = left(index);
    unsigned int r = right(index);
    unsigned int largest = index;

    if (l < vector.getSize() && cmp(vector[l],vector[index]))
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