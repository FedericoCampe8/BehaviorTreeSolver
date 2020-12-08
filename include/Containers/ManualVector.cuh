#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cassert>
#include <type_traits>


#include <thrust/copy.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include <Utils/Memory.cuh>

template<typename T>
class ManualVector
{
    public:
        uint32_t const capacity;
        uint32_t validPrefixSize : 31;
    private:
        bool const owning : 1;
        T * const storage;

    public:
        __host__ __device__ ManualVector(unsigned int capacity);
        __host__ __device__ ManualVector(unsigned int capacity, std::byte* storage);
        __host__ __device__ ~ManualVector();
        __host__ __device__ T& operator[](unsigned int index) const;
        __host__ __device__ ManualVector<T>& operator=(ManualVector<T> const & other);
        __host__ __device__ bool operator==(ManualVector<T> const & other) const;
        __host__ __device__ bool isEmptyValid() const;
        __host__ __device__ bool isFullValid() const;
        __host__ __device__ T& at(unsigned int index) const;
        __host__ __device__ T& firstValid() const;
        __host__ __device__ T& lastValid() const;
        __host__ __device__ T* beginValid() const;
        __host__ __device__ T* endValid() const;
        template<typename ... Args>
        __host__ __device__ void emplaceBackValid(Args ... args);
        __host__ __device__ void pushBackValid(T const & t);
        __host__ __device__ void popBackValid();
        __host__ __device__ void clearValid();
        __host__ __device__ void print(bool endline = true, bool compact = true) const;
        __host__ __device__ std::byte* getStorageEnd(size_t const alignment = 1) const;
        __host__ __device__ static std::size_t sizeofStorage(unsigned int capacity);
    private:
        __host__ __device__ T* mallocStorage();
};

template<typename T>
__host__ __device__
ManualVector<T>::ManualVector(unsigned int capacity) :
    capacity(capacity),
    owning(true),
    validPrefixSize(0),
    storage(mallocStorage())
{}

template<typename T>
__host__ __device__
ManualVector<T>::ManualVector(unsigned int capacity, std::byte* storage) :
    capacity(capacity),
    owning(false),
    validPrefixSize(0),
    storage(reinterpret_cast<T*>(storage))
{}

template<typename T>
__host__ __device__
ManualVector<T>::~ManualVector()
{
    if(owning)
    {
        std::free(storage);
    }
}

template<typename T>
__host__ __device__
T& ManualVector<T>::operator[](unsigned int index) const
{
    assert(index < capacity);
    return storage[index];
}

template<typename T>
__host__ __device__
ManualVector<T>& ManualVector<T>::operator=(ManualVector<T> const & other)
{
    assert(other.validPrefixSize <= capacity);
    validPrefixSize = other.validPrefixSize;
    thrust::copy(thrust::seq, other.beginValid(), other.endValid(), beginValid());
    return *this;
}

template<typename T>
__host__ __device__
bool ManualVector<T>::operator==(ManualVector<T> const & other) const
{
    return validPrefixSize == other.validPrefixSize and thrust::equal(beginValid(), endValid(), other.beginValid());
}

template<typename T>
__host__ __device__
std::size_t ManualVector<T>::sizeofStorage(unsigned int capacity)
{
    return sizeof(T) * capacity;
}

template<typename T>
__host__ __device__
bool ManualVector<T>::isEmptyValid() const
{
    return validPrefixSize == 0;
}

template<typename T>
__host__ __device__
bool ManualVector<T>::isFullValid() const
{
    return validPrefixSize == capacity;
}

template<typename T>
__host__ __device__
T& ManualVector<T>::at(unsigned int index) const
{
    return operator[](index);
}

template<typename T>
__host__ __device__
T& ManualVector<T>::firstValid() const
{
    return operator[](0);
}

template<typename T>
__host__ __device__
T& ManualVector<T>::lastValid() const
{
    return operator[](validPrefixSize - 1);
}

template<typename T>
__host__ __device__
T* ManualVector<T>::beginValid() const
{
    return storage;
}

template<typename T>
__host__ __device__
T* ManualVector<T>::endValid() const
{
    return storage + validPrefixSize;
}

template<typename T>
template<typename ... Args>
__host__ __device__
void ManualVector<T>::emplaceBackValid(Args ... args)
{
    assert(validPrefixSize < capacity);
    new (&storage[validPrefixSize]) T(args ...);
    validPrefixSize += 1;
}

template<typename T>
__host__ __device__
void ManualVector<T>::pushBackValid(T const & t)
{
    assert(validPrefixSize < capacity);
    storage[validPrefixSize] = t;
    validPrefixSize += 1;
}

template<typename T>
__host__ __device__
void ManualVector<T>::clearValid()
{
    validPrefixSize = 0;
}

template<typename T>
__host__ __device__
void ManualVector<T>::popBackValid()
{
    assert(validPrefixSize > 0);
    validPrefixSize -= 1;
}

template<typename T>
__host__ __device__
void ManualVector<T>::print(bool endline, bool compact) const
{
    static_assert(std::is_integral<T>::value);

    printf("[");
    if(validPrefixSize > 0)
    {
        printf("%d",storage[0]);
        for (uint i = 1; i < validPrefixSize; i += 1)
        {
            printf(",%d", storage[i]);
        }
        if (not compact)
        {
            for (uint i = validPrefixSize; i < capacity; i += 1)
            {
                printf(",_");
            }
        }
    }
    printf("]");
    if (endline)
    {
        printf("\n");
    }
}

template<typename T>
__host__ __device__
T* ManualVector<T>::mallocStorage()
{
    std::byte* storage = Memory::safeMalloc(sizeofStorage(capacity));
    return static_cast<T*>(storage);
}

template<typename T>
__host__ __device__
std::byte* ManualVector<T>::getStorageEnd(size_t const alignment) const
{
    return Memory::align(alignment,  reinterpret_cast<std::byte*>(storage + capacity));
}