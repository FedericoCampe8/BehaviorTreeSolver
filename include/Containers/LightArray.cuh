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
        std::size_t capacity;
        T * storage;

    // Functions
    public:
        __host__ __device__ LightArray(std::size_t capacity, T* storage);
        __host__ __device__ ~LightArray();
        __host__ __device__ inline T* at(std::size_t index) const;
        __host__ __device__ inline T* begin() const;
        __host__ __device__ virtual inline T* end() const;
        __host__ __device__ inline std::size_t getCapacity() const;
        __host__ __device__ virtual inline std::size_t indexOf(T const * t) const;
        __host__ __device__ void operator=(LightArray<T> const & other);
        __host__ __device__ virtual inline T* operator[](std::size_t index) const;
        __host__ __device__ virtual void print(bool endLine = true) const;
        __host__ __device__ static std::size_t sizeOfStorage(std::size_t capacity);
        __host__ __device__ static void swap(LightArray<T>* a0, LightArray<T>* a1);
    protected:
        __host__ __device__ void print(std::size_t beginIdx, std::size_t endIdx, bool endLine) const;

};

template<typename T>
__host__ __device__
LightArray<T>::LightArray(std::size_t capacity, T* storage) :
    capacity(capacity),
    storage(storage)
{}

template<typename T>
__host__ __device__
LightArray<T>::~LightArray()
{}

template<typename T>
__host__ __device__
T* LightArray<T>::at(std::size_t index) const
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
std::size_t LightArray<T>::getCapacity() const
{
    return capacity;
}

template<typename T>
__host__ __device__
std::size_t LightArray<T>::indexOf(T const * t) const
{
    assert(begin() <= t);
    assert(t < end());

    return thrust::distance(const_cast<T const *>(begin()), t);
}

template<typename T>
__host__ __device__
void LightArray<T>::operator=(LightArray<T> const & other)
{
    this->capacity = other.capacity;
    this->storage = other.storage;
}

template<typename T>
__host__ __device__
T* LightArray<T>::operator[](std::size_t index) const
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
std::size_t LightArray<T>::sizeOfStorage(std::size_t capacity)
{
    return sizeof(T) * capacity;
}

template<typename T>
__host__ __device__
void LightArray<T>::swap(LightArray<T>* a0, LightArray<T>* a1)
{
    thrust::swap(a0->capacity, a1->capacity);
    thrust::swap(a0->storage, a1->storage);
}

template<typename T>
__host__ __device__
void LightArray<T>::print(std::size_t beginIdx, std::size_t endIdx, bool endLine) const
{
    if constexpr (std::is_integral_v<T>)
    {
        printf("[");
        if (beginIdx < endIdx)
        {
            printf("%d", storage[beginIdx]);
            for (std::size_t index = beginIdx + 1; index < endIdx; index += 1)
            {
                printf(",%d", storage[index]);
            }
        }
        printf("]");
        if (endLine)
        {
            printf("\n");
        }
    }
}