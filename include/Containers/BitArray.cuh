#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <thrust/copy.h>
#include <thrust/swap.h>
#include <Utils/Memory.cuh>

template<typename T>
class BitArray
{
    // Members
    protected:
    uint32_t capacity;
    T * storage;

    // Functions
    public:
    __host__ __device__ BitArray(unsigned int capacity, T* storage);
    __host__ __device__ ~BitArray();
    __host__ __device__ inline T* at(unsigned int index) const;
    __host__ __device__ inline thrust::detail::seq_t begin() const;
    __host__ __device__ inline T* end() const;
    __host__ __device__ inline std::byte* endOfStorage() const;
    __host__ __device__ inline unsigned int getCapacity() const;
    __host__ __device__ inline unsigned int indexOf(T const * t) const;
    __host__ __device__ BitArray<T>& operator=(BitArray<T> const & other);
    __host__ __device__ inline T* operator[](unsigned int index) const;
    __host__ __device__ void print(bool endLine = true) const;
    __host__ __device__ inline static unsigned int sizeOfStorage(unsigned int capacity);
    __host__ __device__ inline static void swap(BitArray<T>& a0, BitArray<T>& a1);
    protected:
    __host__ __device__ void print(unsigned int beginIdx, unsigned int endIdx, bool endLine) const;

};

template<typename T>
__host__ __device__
BitArray<T>::BitArray(unsigned int capacity, T* storage) :
    capacity(capacity),
    storage(storage)
{}

template<typename T>
__host__ __device__
BitArray<T>::~BitArray()
{}

template<typename T>
__host__ __device__
T* BitArray<T>::at(unsigned int index) const
{
    assert(capacity > 0);
    assert(index < capacity);
    return storage + index;
}

template<typename T>
__host__ __device__
thrust::detail::seq_t BitArray<T>::begin() const
{
    return storage;
}

template<typename T>
__host__ __device__
T* BitArray<T>::end() const
{
    return storage + capacity;
}

template<typename T>
__host__ __device__
std::byte* BitArray<T>::endOfStorage() const
{
    return reinterpret_cast<std::byte*>(storage + capacity);
}

template<typename T>
__host__ __device__
unsigned int BitArray<T>::getCapacity() const
{
    return capacity;
}

template<typename T>
__host__ __device__
unsigned int BitArray<T>::indexOf(T const * t) const
{
    T const * const begin = this->begin();
    assert(begin <= t);
    assert(t < end());
    return static_cast<unsigned int>(t - begin);
}

template<typename T>
__host__ __device__
BitArray<T>& BitArray<T>::operator=(BitArray<T> const & other)
{
    capacity = other.capacity;
    storage = other.storage;
    return *this;
}

template<typename T>
__host__ __device__
T* BitArray<T>::operator[](unsigned int index) const
{
    return at(index);
}

template<typename T>
__host__ __device__
void BitArray<T>::print(bool endLine) const
{
    print(0, capacity, endLine);
}

template<typename T>
__host__ __device__
unsigned int BitArray<T>::sizeOfStorage(unsigned int capacity)
{
    return sizeof(T) * capacity;
}

template<typename T>
__host__ __device__
void BitArray<T>::swap(BitArray<T>& a0, BitArray<T>& a1)
{
    thrust::swap(a0.capacity, a1.capacity);
    thrust::swap(a0.storage, a1.storage);
}

template<typename T>
__host__ __device__
void BitArray<T>::print(unsigned int beginIdx, unsigned int endIdx, bool endLine) const
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