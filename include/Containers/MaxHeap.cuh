#pragma once

#include <thrust/swap.h>
#include <thrust/distance.h>

template<typename T>
class MaxHeap
{
    public:
        typedef bool (*Comparator) (T const & t0, T const & t1);

    private:
        Comparator const cmp;
        Vector<T> vector;

    public:
        __host__ __device__ MaxHeap(Comparator cmp, unsigned int capacity, Memory::MallocType mallocType);
        template<typename ... Args>
        __host__ __device__ void emplaceBack(Args ... args);
        __host__ __device__ void erase(unsigned int index);
        __host__ __device__ inline T& front() const;
        __host__ __device__ inline unsigned int getSize();
        __host__ __device__ inline unsigned int getCapacity();
        __host__ __device__ void insert();
        __host__ __device__ inline unsigned int isEmpty();
        __host__ __device__ void popBack();

    private:
        __host__ __device__ inline unsigned int parent(unsigned int index);
        __host__ __device__ inline unsigned int left(unsigned int index);
        __host__ __device__ inline unsigned int right(unsigned int index);
        __host__ __device__ void heapify(unsigned int index);
};

template<typename T>
__host__ __device__
MaxHeap<T>::MaxHeap(Comparator cmp, unsigned int capacity, Memory::MallocType mallocType) :
    cmp(cmp),
    vector(capacity, mallocType)
{}

template<typename T>
template<typename ... Args>
__host__ __device__
void MaxHeap<T>::emplaceBack(Args ... args)
{
    vector.emplaceBack(args ...);
}

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
unsigned int MaxHeap<T>::getCapacity()
{
    return vector.getCapacity();
}

template<typename T>
__host__ __device__
void MaxHeap<T>::insert()
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
unsigned int MaxHeap<T>::isEmpty()
{
    return vector.isEmpty();
}

template<typename T>
__host__ __device__
void MaxHeap<T>::popBack()
{
    vector.popBack();
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