#pragma once

#include <thrust/sequence.h>
#include <Containers/Array.cuh>
#include <Containers/Vector.cuh>

template<typename T>
class Buffer
{
    // Members
    private:
    Array<T> buffer;
    Vector<unsigned int> invalids;

    // Functions
    public:
    __host__ __device__ Buffer(unsigned int capacity, Memory::MallocType mallocType);
    __host__ __device__ inline T* at(unsigned int index) const;
    __host__ __device__ void clear();
    __host__ __device__ inline void erase(T const * t);
    __host__ __device__ inline unsigned int getCapacity() const;
    __host__ __device__ inline unsigned int getSize() const;
    __host__ __device__ inline unsigned int indexOf(T const * t);
    __host__ __device__ T* insert(T const * t);
    __host__ __device__ inline bool isEmpty() const;
    __host__ __device__ inline bool isFull() const;
    __host__ __device__ T* operator[](unsigned int index) const;
};

template<typename T>
__host__ __device__
Buffer<T>::Buffer(unsigned int capacity, Memory::MallocType mallocType) :
    buffer(capacity, mallocType),
    invalids(capacity, mallocType)
{
    clear();
}

template<typename T>
__host__ __device__
T* Buffer<T>::at(unsigned int index) const
{
    return buffer[index];
}

template<typename T>
__host__ __device__
void Buffer<T>::clear()
{
    invalids.resize(invalids.getCapacity());
    thrust::sequence(thrust::seq, invalids.begin(), invalids.end());
}

template<typename T>
__host__ __device__
void Buffer<T>::erase(T const * t)
{
    unsigned int const invalidIdx = buffer.indexOf(t);
    invalids.pushBack(&invalidIdx);
}

template<typename T>
__host__ __device__
unsigned int Buffer<T>::getCapacity() const
{
    return buffer.getCapacity();
}


template<typename T>
__host__ __device__
unsigned int Buffer<T>::getSize() const
{
    return invalids.getCapacity() - invalids.getSize();
}

template<typename T>
__host__ __device__
unsigned int Buffer<T>::indexOf(T const * t)
{
    return buffer.indexOf(t);
}

template<typename T>
__host__ __device__
T* Buffer<T>::insert(T const * t)
{
    assert(not isFull());
    unsigned int const elementIdx = *invalids.back();
    invalids.popBack();
    *at(elementIdx) = *t;
    return at(elementIdx);
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
T* Buffer<T>::operator[](unsigned int index) const
{
    return at(index);
}