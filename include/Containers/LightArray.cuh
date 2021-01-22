#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/swap.h>
#include <Utils/Memory.cuh>

template<typename T>
class LightArray
{
    // Members
    protected:
    unsigned int capacity;
    T * storage;

    // Functions
    public:
    __host__ __device__ LightArray(unsigned int capacity, T* storage);
    __host__ __device__ ~LightArray();
    __host__ __device__ inline T* at(unsigned int index) const;
    __host__ __device__ inline T* begin() const;
    __host__ __device__ inline T* end() const;
    __host__ __device__ inline std::byte* endOfStorage() const;
    __host__ __device__ inline unsigned int getCapacity() const;
    __host__ __device__ inline unsigned int indexOf(T const * t) const;
    __host__ __device__ LightArray<T>& operator=(LightArray<T> const & other);
    __host__ __device__ inline T* operator[](unsigned int index) const;
    __host__ __device__ void print(bool endLine = true) const;
    __host__ __device__ inline static unsigned int sizeOfStorage(unsigned int capacity);
    __host__ __device__ inline static void swap(LightArray<T>& a0, LightArray<T>& a1);
    protected:
    __host__ __device__ void print(unsigned int beginIdx, unsigned int endIdx, bool endLine) const;

};

template<typename T>
__host__ __device__
LightArray<T>::LightArray(unsigned int capacity, T* storage) :
    capacity(capacity),
    storage(storage)
{}

template<typename T>
__host__ __device__
LightArray<T>::~LightArray()
{}

template<typename T>
__host__ __device__
T* LightArray<T>::at(unsigned int index) const
{
    assert(capacity > 0);
    assert(index < capacity);
    return storage + index;
}

template<typename T>
__host__ __device__
T* LightArray<T>::begin() const
{
    return storage;
}

template<typename T>
__host__ __device__
T* LightArray<T>::end() const
{
    return storage + capacity;
}

template<typename T>
__host__ __device__
std::byte* LightArray<T>::endOfStorage() const
{
    return reinterpret_cast<std::byte*>(storage + capacity);
}

template<typename T>
__host__ __device__
unsigned int LightArray<T>::getCapacity() const
{
    return capacity;
}

template<typename T>
__host__ __device__
unsigned int LightArray<T>::indexOf(T const * t) const
{
    T const * const b = begin();
    assert(b <= t);
    assert(t < end());
    return static_cast<unsigned int>(thrust::distance(b, t));
}

template<typename T>
__host__ __device__
LightArray<T>& LightArray<T>::operator=(LightArray<T> const & other)
{
    capacity = other.capacity;
    storage = other.storage;
    return *this;
}

template<typename T>
__host__ __device__
T* LightArray<T>::operator[](unsigned int index) const
{
    return at(index);
}

template<typename T>
__host__ __device__
void LightArray<T>::print(bool endLine) const
{
    print(0, capacity, endLine);
}

template<typename T>
__host__ __device__
unsigned int LightArray<T>::sizeOfStorage(unsigned int capacity)
{
    return sizeof(T) * capacity;
}

template<typename T>
__host__ __device__
void LightArray<T>::swap(LightArray<T>& a0, LightArray<T>& a1)
{
    thrust::swap(a0.capacity, a1.capacity);
    thrust::swap(a0.storage, a1.storage);
}

template<typename T>
__host__ __device__
void LightArray<T>::print(unsigned int beginIdx, unsigned int endIdx, bool endLine) const
{
    if constexpr (std::is_integral_v<T>)
    {
        printf("[");
        if (beginIdx < endIdx)
        {
            printf("%d",static_cast<int>(*at(beginIdx)));
            for (unsigned int index = beginIdx + 1; index < endIdx; index += 1)
            {
                printf(",");
                printf("%d", static_cast<int>(*at(index)));
            }
        }
        printf(endLine ? "]\n" : "]");
    }
}