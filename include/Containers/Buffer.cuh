#pragma once

#include <cstdint>
#include <utility>
#include <thrust/distance.h>
#include <thrust/sequence.h>
#include <Containers/Array.cuh>
#include <Containers/Vector.cuh>
#include <Utils/Memory.cuh>

template<typename T>
class Buffer
{
    private:
        Array<T> elements;
        Vector<uint32_t> invalids;

    public:
        __host__ __device__ Buffer(unsigned int capacity, Memory::MallocType mallocType);
        __host__ __device__ inline T& at(unsigned int index) const;
        __host__ __device__ inline T* begin() const;
        __host__ __device__ void clear();
        __host__ __device__ inline T* end() const;
        __host__ __device__ inline void erase(T const * t);
        __host__ __device__ inline unsigned int getCapacity() const;
        __host__ __device__ inline unsigned int getSize() const;
        __host__ __device__ inline unsigned int indexOf(T& t) const;
        __host__ __device__ T& insert(T const & t);
        __host__ __device__ inline bool isEmpty() const;
        __host__ __device__ inline bool isFull() const;
        __host__ __device__ inline T& operator[](unsigned int index) const;
};

template<typename T>
__host__ __device__
Buffer<T>::Buffer(unsigned int capacity, Memory::MallocType mallocType) :
    elements(capacity, mallocType),
    invalids(capacity, mallocType)
{
    clear();
}

template<typename T>
__host__ __device__
T& Buffer<T>::at(unsigned int index) const
{
    return elements.at(index);
}

template<typename T>
__host__ __device__
T* Buffer<T>::begin() const
{
    return elements.begin();
}

template<typename T>
__host__ __device__
void Buffer<T>::clear()
{
    unsigned int capacity = elements.getCapacity();
    invalids.resize(capacity);
    for(unsigned int i = 0; i < capacity; i += 1)
    {
        invalids.at(i) = capacity - 1 - i;
    }
}

template<typename T>
__host__ __device__
T* Buffer<T>::end() const
{
    return elements.end();
}

template<typename T>
__host__ __device__
void Buffer<T>::erase(T const * t)
{
    assert(elements.begin() <= t);
    assert(t < elements.end());

    unsigned int const elementIdx = thrust::distance(elements.begin(), t);
    invalids.pushBack(elementIdx);
}

template<typename T>
__host__ __device__
unsigned int Buffer<T>::getCapacity() const
{
    return elements.getCapacity();
}

template<typename T>
__host__ __device__
unsigned int Buffer<T>::getSize() const
{
    return elements.getCapacity() - invalids.getSize();
}

template<typename T>
__host__ __device__
unsigned int Buffer<T>::indexOf(T& t) const
{
    assert(elements.begin() <= &t);
    assert(&t < elements.end());
    return thrust::distance(elements.begin(),&t);
}

template<typename T>
__host__ __device__
T& Buffer<T>::insert(T const & t)
{
    assert(not isFull());

    unsigned int const elementIdx = invalids.back();
    invalids.popBack();
    elements.at(elementIdx) = t;
    return elements.at(elementIdx);
}

template<typename T>
__host__ __device__
bool Buffer<T>::isEmpty() const
{
    return invalids.isFull();
}

template<typename T>
__host__ __device__
bool Buffer<T>::isFull() const
{
    return invalids.isEmpty();
}

template<typename T>
__host__ __device__
T& Buffer<T>::operator[](unsigned int index) const
{
    return elements[index];
}