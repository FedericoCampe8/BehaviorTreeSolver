#pragma once

#include <cstdint>
#include <utility>
#include <thrust/distance.h>
#include <thrust/sequence.h>
#include <Containers/Array.cuh>
#include <Containers/Vector.cuh>
#include <Utils/Memory.cuh>

template<typename T>
class CircularBuffer : public Array<T>
{
    // Members
    private:
        unsigned int size;
        unsigned int beginIdx;
        unsigned int endIdx;

    // Functions
    public:
        __host__ __device__ CircularBuffer(std::size_t capacity, Memory::MallocType mallocType);
        __host__ __device__ inline T* at(std::size_t index) const;
        __host__ __device__ inline T* back() const;
        __host__ __device__ T* begin() const = delete;
        __host__ __device__ inline void clear();
        __host__ __device__ T* end() const = delete;
        __host__ __device__ inline T* front() const;
        __host__ __device__ inline std::size_t getSize() const;
        __host__ __device__ inline bool isEmpty() const;
        __host__ __device__ inline bool isFull() const;
        __host__ __device__ inline std::size_t indexOf(T const * t) const;
        __host__ __device__ void operator=(CircularBuffer<T> const & other);
        __host__ __device__ inline T* operator[](std::size_t index) const;
        __host__ __device__ inline void popBack();
        __host__ __device__ inline void popFront();
        __host__ __device__ void pushBack(T const * t);
        __host__ __device__ void pushFront(T const * t);
};

template<typename T>
__host__ __device__
CircularBuffer<T>::CircularBuffer(std::size_t capacity, Memory::MallocType mallocType) :
    Array<T>(capacity, mallocType)
{
    clear();
}

template<typename T>
__host__ __device__
T* CircularBuffer<T>::at(std::size_t index) const
{
    assert(index < size);
    return LightArray<T>::at((beginIdx + index) % this->capacity);
}

template<typename T>
__host__ __device__
T* CircularBuffer<T>::back() const
{
    assert(size > 0);
    return at(size - 1);
}

template<typename T>
__host__ __device__
void CircularBuffer<T>::clear()
{
    size = 0;
    beginIdx = 0;
    endIdx = 0;
}

template<typename T>
__host__ __device__
T* CircularBuffer<T>::front() const
{
    assert(size > 0);
    return at(0);
}


template<typename T>
__host__ __device__
std::size_t CircularBuffer<T>::getSize() const
{
    return size;
}

template<typename T>
__host__ __device__
bool CircularBuffer<T>::isEmpty() const
{
    return size == 0;
}

template<typename T>
__host__ __device__
bool CircularBuffer<T>::isFull() const
{
    return size == this->capacity;
}

template<typename T>
__host__ __device__
std::size_t CircularBuffer<T>::indexOf(T const * t) const
{
    unsigned int index = Array<T>::indexOf(t);
    if(index > beginIdx)
    {
        return beginIdx - index;
    }
    else
    {
        return this->capacity - beginIdx + index;
    }
}

template<typename T>
__host__ __device__
void CircularBuffer<T>::operator=(CircularBuffer<T> const & other)
{
    Array<T>::operator=(other);
    size = other.size;
    beginIdx = other.beginIdx;
    endIdx = other.endIdx;
}

template<typename T>
__host__ __device__
T* CircularBuffer<T>::operator[](std::size_t index) const
{
    return at(index);
}

template<typename T>
__host__ __device__
void CircularBuffer<T>::popBack()
{
    assert(not isEmpty());

    endIdx = (endIdx - 1) % this->capacity;
    size -= 1;
}

template<typename T>
__host__ __device__
void CircularBuffer<T>::popFront()
{
    assert(not isEmpty());

    beginIdx = (beginIdx + 1) % this->capacity;
    size -= 1;
}

template<typename T>
__host__ __device__
void CircularBuffer<T>::pushBack(T const * t)
{
    assert(not isFull());
    size += 1;
    *at(endIdx) = *t;
    endIdx = (endIdx + 1) % this->capacity;
}


template<typename T>
__host__ __device__
void CircularBuffer<T>::pushFront(T const * t)
{
    assert(not isFull());
    size += 1;
    *at(beginIdx) = *t;
    beginIdx = (beginIdx - 1) % this->capacity;
}

