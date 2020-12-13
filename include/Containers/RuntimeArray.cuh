#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cassert>
#include <type_traits>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>

#include <Utils/Memory.cuh>

template<typename T>
class RuntimeArray
{
    private:
        enum Flag {Owning = 1};
        unsigned int flags;
        unsigned int capacity;
        T * storage;

    public:
        __host__ __device__ RuntimeArray(unsigned int capacity, Memory::MallocType allocType = Memory::MallocType::Std);
        __host__ __device__ RuntimeArray(unsigned int capacity, std::byte* storage);
        __host__ __device__ RuntimeArray(T* begin, T* end);
        __host__ __device__ ~RuntimeArray();
        __host__ __device__ inline T& at(unsigned int index) const;
        __host__ __device__ inline T& back() const;
        __host__ __device__ inline T* begin() const;
        __host__ __device__ inline T* end() const;
        __host__ __device__ inline T& front() const;
        __host__ __device__ inline unsigned int getCapacity() const;
    private:
        __host__ __device__ static T* mallocStorage(unsigned int capacity, Memory::MallocType allocType);
    public:
        __host__ __device__ RuntimeArray<T>& operator=(RuntimeArray<T> const & other);
        __host__ __device__ inline T& operator[](unsigned int index) const;
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ static inline std::size_t sizeOfStorage(unsigned int capacity);
        __host__ __device__ inline std::byte* storageEnd() const;
        __host__ __device__ void swap(RuntimeArray<T>& other);
};

template<typename T>
__host__ __device__
RuntimeArray<T>::RuntimeArray(unsigned int capacity, Memory::MallocType allocType) :
    flags(Flag::Owning),
    capacity(capacity),
    storage(mallocStorage(capacity, allocType))
{}

template<typename T>
__host__ __device__
RuntimeArray<T>::RuntimeArray(unsigned int capacity, std::byte* storage) :
    flags(0),
    capacity(capacity),
    storage(reinterpret_cast<T*>(storage))
{}

template<typename T>
__host__ __device__
RuntimeArray<T>::RuntimeArray(T* begin, T* end) :
    flags(0),
    capacity(thrust::distance(begin,end)),
    storage(begin)
{}

template<typename T>
__host__ __device__
RuntimeArray<T>::~RuntimeArray()
{
    if(flags & Flag::Owning)
    {
        free(storage);
    }
}

template<typename T>
__host__ __device__
T& RuntimeArray<T>::at(unsigned int index) const
{
    assert(capacity > 0);
    assert(index < capacity);

    return storage[index];
}

template<typename T>
__host__ __device__
T& RuntimeArray<T>::back() const
{
    return at(capacity - 1);
}

template<typename T>
__host__ __device__
T* RuntimeArray<T>::begin() const
{
    return storage;
}

template<typename T>
__host__ __device__
T* RuntimeArray<T>::end() const
{
    return storage + capacity;
}

template<typename T>
__host__ __device__
T& RuntimeArray<T>::front() const
{
    return at(0);
}

template<typename T>
__host__ __device__
unsigned int RuntimeArray<T>::getCapacity() const
{
    return capacity;
}

template<typename T>
__host__ __device__
T* RuntimeArray<T>::mallocStorage(unsigned int capacity, Memory::MallocType allocType)
{
    std::byte* storage = Memory::safeMalloc(sizeOfStorage(capacity), allocType);
    return reinterpret_cast<T*>(storage);
}

template<typename T>
__host__ __device__
RuntimeArray<T>& RuntimeArray<T>::operator=(RuntimeArray<T> const & other)
{
    assert(capacity >= other.capacity);
    thrust::copy(
#ifdef __CUDA_ARCH__
            thrust::device,
#else
            thrust::host,
#endif
            other.begin(), other.end(), begin());
    return *this;
}

template<typename T>
__host__ __device__
T& RuntimeArray<T>::operator[](unsigned int index) const
{
    return at(index);
}

template<typename T>
__host__ __device__
void RuntimeArray<T>::print(bool endLine) const
{
    static_assert(std::is_integral<T>::value);

    printf("[");
    if(capacity > 0)
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
std::size_t RuntimeArray<T>::sizeOfStorage(unsigned int capacity)
{
    return sizeof(T) * capacity;
}

template<typename T>
__host__ __device__
std::byte* RuntimeArray<T>::storageEnd() const
{
    return reinterpret_cast<std::byte*>(storage + capacity);
}

template<typename T>
__host__ __device__
void RuntimeArray<T>::swap(RuntimeArray<T>& other)
{
    thrust::swap(flags, other.flags);
    thrust::swap(capacity, other.capacity);
    thrust::swap(storage, other.storage);
}