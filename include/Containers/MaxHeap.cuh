#pragma once

#include <thrust/swap.h>
#include <thrust/distance.h>

template<typename T>
class MaxHeap : public Vector<T>
{
    // Aliases, Enums, ...
    typedef bool (*Comparator)(T const & t0, T const & t1);

    // Members
    private:
        Comparator const cmp;

    // Functions
    public:
        __host__ __device__ MaxHeap(Comparator cmp, unsigned int capacity, Memory::MallocType mallocType);
        __host__ __device__ void erase(T const * t);
        __host__ __device__ void insertBack();
    private:
        __host__ __device__ void heapify(unsigned int index);
        __host__ __device__ inline unsigned int left(unsigned int index);
        __host__ __device__ inline unsigned int parent(unsigned int index);
        __host__ __device__ inline unsigned int right(unsigned int index);
};

template<typename T>
__host__ __device__
MaxHeap<T>::MaxHeap(Comparator cmp, unsigned int capacity, Memory::MallocType mallocType) :
    Vector<T>(capacity, mallocType),
    cmp(cmp)
{}

template<typename T>
__host__ __device__
void MaxHeap<T>::erase(T const * t)
{
    unsigned int i = this->indexOf(t);
    unsigned int p = parent(i);
    while (i > 0)
    {
        T::swap(this->at(i), this->at(p));
        i = p;
        p = parent(i);
    }
    T::swap(this->front(), this->back());
    this->popBack();
    heapify(0);
}

template<typename T>
__host__ __device__
void MaxHeap<T>::insertBack()
{
    unsigned int i = this->size - 1;
    unsigned int p = parent(i);
    while (i > 0 and (not cmp(*this->at(p), *this->at(i))))
    {
        T::swap(this->at(p), this->at(i));
        i = p;
        p = parent(i);
    }
}

template<typename T>
__host__ __device__
void MaxHeap<T>::heapify(unsigned int index)
{
    unsigned int const l = left(index);
    unsigned int const r = right(index);
    unsigned int largest = index;

    if (l < this->size and cmp(*this->at(l), *this->at(index)))
    {
        largest = l;
    }

    if (r < this->size and cmp(*this->at(r), *this->at(largest)))
    {
        largest = r;
    }

    if (largest != index)
    {
        T::swap(this->at(index), this->at(largest));
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