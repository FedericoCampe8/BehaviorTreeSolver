#pragma once

#include <cstdint>
#include <utility>
#include <thrust/distance.h>
#include <thrust/sequence.h>
#include <Containers/Array.cuh>
#include <Containers/Vector.cuh>
#include <Utils/Memory.cuh>

template<typename T>
class Buffer : public Array<T>
{
    // Members
    private:
        Vector<uint32_t> invalids;

    // Functions
    public:
        __host__ __device__ Buffer(std::size_t capacity, Memory::MallocType mallocType);
        __host__ __device__ void clear();
        __host__ __device__ inline void erase(T const * t);
        __host__ __device__ inline std::size_t getSize() const;
        __host__ __device__ T* insert(T const * t);
        __host__ __device__ inline bool isEmpty() const;
        __host__ __device__ inline bool isFull() const;
};

template<typename T>
__host__ __device__
Buffer<T>::Buffer(std::size_t capacity, Memory::MallocType mallocType) :
    Array<T>(capacity, mallocType),
    invalids(capacity, mallocType)
{
    clear();
}

template<typename T>
__host__ __device__
void Buffer<T>::clear()
{
    invalids.resize(this->capacity);
    for(unsigned int i = 0; i < this->capacity; i += 1)
    {
        *invalids.at(i) = this->capacity - 1 - i;
    }
}

template<typename T>
__host__ __device__
void Buffer<T>::erase(T const * t)
{
    assert(this->begin() <= t);
    assert(t < this->end());

    unsigned int invalidIdx = this->indexOf(t);
    //printf("Erasing %d\n", invalidIdx);
    invalids.incrementSize();
    *invalids.back() = invalidIdx;
}

template<typename T>
__host__ __device__
std::size_t Buffer<T>::getSize() const
{
    return invalids.getCapacity() - invalids.getSize();
}

template<typename T>
__host__ __device__
T* Buffer<T>::insert(T const * t)
{
    assert(not isFull());

    unsigned int const elementIdx = *invalids.back();
    invalids.popBack();
    *this->at(elementIdx) = *t;
    return this->at(elementIdx);
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