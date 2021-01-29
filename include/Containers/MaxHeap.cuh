#pragma once

#include "Containers/Vector.cuh"
template<typename T>
class MaxHeap
{
    // Members
    private:
    Vector<T> vector;

    // Functions
    public:
    __host__ __device__ MaxHeap(unsigned int capacity, Memory::MallocType mallocType);
    __host__ __device__ void erase(T const * t);
    __host__ __device__ inline T* front() const;
    __host__ __device__ void insert(T const * t);
    __host__ __device__ inline bool isEmpty() const;
    __host__ __device__ inline bool isFull() const;
    private:
    __host__ __device__ void heapify(unsigned int index);
    __host__ __device__ inline unsigned int left(unsigned int index);
    __host__ __device__ inline unsigned int parent(unsigned int index);
    __host__ __device__ inline unsigned int right(unsigned int index);
};

template<typename T>
__host__ __device__
MaxHeap<T>::MaxHeap(unsigned int capacity, Memory::MallocType mallocType) :
    vector(capacity, mallocType)
{}

template<typename T>
__host__ __device__
void MaxHeap<T>::erase(T const * t)
{
    unsigned int i = vector.indexOf(t);
    unsigned int p = parent(i);
    while (i > 0)
    {
        T::swap(*vector[i], *vector[p]);
        i = p;
        p = parent(i);
    }
    T::swap(*vector.front(), *vector.back());
    vector.popBack();
    heapify(0);
}

template<typename T>
__host__ __device__
T* MaxHeap<T>::front() const
{
    return vector.front();
}

template<typename T>
__host__ __device__
void MaxHeap<T>::insert(T const * t)
{
    vector.pushBack(t);
    unsigned int i = vector.getSize() - 1;
    unsigned int p = parent(i);
    while (i > 0 and (not (*vector[p] < *vector[i])))
    {
        T::swap(*vector[p], *vector[i]);
        i = p;
        p = parent(i);
    }
}

template<typename T>
__host__ __device__
bool MaxHeap<T>::isEmpty() const
{
    return vector.isEmpty();
}

template<typename T>
__host__ __device__
bool MaxHeap<T>::isFull() const
{
    return vector.isFull();
}

template<typename T>
__host__ __device__
void MaxHeap<T>::heapify(unsigned int index)
{
    unsigned int const l = left(index);
    unsigned int const r = right(index);
    unsigned int largest = index;
    if (l < vector.getSize() and (*vector[l] < *vector[index]))
    {
        largest = l;
    }
    if (r < vector.getSize() and (*vector[r] < *vector[largest]))
    {
        largest = r;
    }
    if (largest != index)
    {
        T::swap(*vector[index], *vector[largest]);
        heapify(largest);
    }
}

template<typename T>
__host__ __device__
unsigned int MaxHeap<T>::left(unsigned int index)
{
    return 2 * index + 1;
}

template<typename T>
__host__ __device__
unsigned int MaxHeap<T>::parent(unsigned int index)
{
    return (index - 1) / 2;
}

template<typename T>
__host__ __device__
unsigned int MaxHeap<T>::right(unsigned int index)
{
    return 2 * index + 2;
}