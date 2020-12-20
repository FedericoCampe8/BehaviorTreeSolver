#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/swap.h>
#include <Utils/Memory.cuh>

template<typename T>
class Array
{
    private:
        enum Flags {Owning = 1};

    private:
        uint32_t flags;
        uint32_t capacity;
        T * storage;

    public:
        // Storage
        __host__ __device__ static std::byte* mallocStorage(unsigned int capacity, Memory::MallocType mallocType);
        __host__ __device__ static std::size_t sizeOfStorage(unsigned int capacity);
        __host__ __device__ std::byte* storageEnd() const;

        // Array
        __host__ __device__ Array(unsigned int capacity, Memory::MallocType mallocType);
        __host__ __device__ Array(unsigned int capacity, std::byte* storage);
        __host__ __device__ Array(T* begin, T* end);
        __host__ __device__ ~Array();
        __host__ __device__ inline T& at(unsigned int index) const;
        __host__ __device__ inline T& back() const;
        __host__ __device__ inline T* begin() const;
        __host__ __device__ inline T* end() const;
        __host__ __device__ inline T& front() const;
        __host__ __device__ inline unsigned int getCapacity() const;
        __host__ __device__ inline unsigned int indexOf(T const * t) const;
        __host__ __device__ Array<T>& operator=(Array<T> const & other);
        __host__ __device__ inline T& operator[](unsigned int index) const;
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ static void swap(Array<T>& a0, Array<T>& a1);
};

template<typename T>
__host__ __device__
std::byte* Array<T>::mallocStorage(unsigned int capacity, Memory::MallocType mallocType)
{
    return  Memory::safeMalloc(sizeOfStorage(capacity), mallocType);
}

template<typename T>
__host__ __device__
std::size_t Array<T>::sizeOfStorage(unsigned int capacity)
{
    return sizeof(T) * capacity;
}

template<typename T>
__host__ __device__
std::byte* Array<T>::storageEnd() const
{
    return reinterpret_cast<std::byte*>(storage + capacity);
}

template<typename T>
__host__ __device__
Array<T>::Array(unsigned int capacity, Memory::MallocType mallocType) :
    flags(Flags::Owning),
    capacity(capacity),
    storage(mallocStorage(capacity,mallocType))
{}

template<typename T>
__host__ __device__
Array<T>::Array(unsigned int capacity, std::byte* storage) :
    flags(0),
    capacity(capacity),
    storage(reinterpret_cast<T*>(storage))
{}

template<typename T>
__host__ __device__
Array<T>::Array(T* begin, T* end) :
    flags(0),
    capacity(thrust::distance(begin, end)),
    storage(end)
{}


template<typename T>
__host__ __device__
Array<T>::~Array()
{
    if(flags & Flags::Owning)
    {
        free(storage);
    }
}

template<typename T>
__host__ __device__
T& Array<T>::at(unsigned int index) const
{
    assert(capacity > 0);
    assert(index < capacity);

    return storage[index];
}

template<typename T>
__host__ __device__
T& Array<T>::back() const
{
    return at(capacity - 1);
}

template<typename T>
__host__ __device__
T* Array<T>::begin() const
{
    return storage;
}

template<typename T>
__host__ __device__
T* Array<T>::end() const
{
    return storage + capacity;
}

template<typename T>
__host__ __device__
T& Array<T>::front() const
{
    return at(0);
}

template<typename T>
__host__ __device__
unsigned int Array<T>::getCapacity() const
{
    return capacity;
}

template<typename T>
__host__ __device__
unsigned int Array<T>::indexOf(T const * t) const
{
    assert(begin() <= t);
    assert(t < end());

    return thrust::distance(begin(), t);
}

template<typename T>
__host__ __device__
T* Array<T>::mallocStorage(unsigned int capacity, Memory::MallocType mallocType)
{
    std::byte* storage = Memory::safeMalloc(sizeOfStorage(capacity), mallocType);
    return reinterpret_cast<T*>(storage);
}

template<typename T>
__host__ __device__
Array<T>& Array<T>::operator=(Array<T> const & other)
{
    assert(capacity >= other.capacity);
    thrust::copy(thrust::seq, other.begin(), other.end(), begin());
    return *this;
}

template<typename T>
__host__ __device__
T& Array<T>::operator[](unsigned int index) const
{
    return at(index);
}

template<typename T>
__host__ __device__
void Array<T>::print(bool endLine) const
{
    static_assert(std::is_integral<T>::value);

    printf("[");
    if (capacity > 0)
    {
        printf("%d", at(0));
        for (unsigned int i = 1; i < capacity; i += 1)
        {
            printf(",%d", at(i));
        }
    }
    printf("]");
    if (endLine)
    {
        printf("\n");
    }
}

template<typename T>
__host__ __device__
void Array<T>::swap(Array<T>& a0, Array<T>& a1)
{
    thrust::swap(a0.flags, a1.flags);
    thrust::swap(a0.capacity, a1.capacity);
    thrust::swap(a0.storage, a1.storage);
}