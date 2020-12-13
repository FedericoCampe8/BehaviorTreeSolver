#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cassert>
#include <type_traits>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/swap.h>

#include <Utils/Memory.cuh>

template<typename T>
class StaticVector
{
    private:
        enum Flag {Owning = 1};
        unsigned int flags;
        unsigned int size;
        unsigned int capacity;
        T * storage;

    public:
        __host__ __device__ StaticVector(unsigned int capacity, Memory::MallocType allocType = Memory::MallocType::Std);
        __host__ __device__ StaticVector(unsigned int capacity, std::byte* storage);
        __host__ __device__ StaticVector(T* begin, T* end);
        __host__ __device__ ~StaticVector();
        __host__ __device__ inline T& at(unsigned int index) const;
        __host__ __device__ inline T& back() const;
        __host__ __device__ inline T* begin() const;
        __host__ __device__ inline void clear();
        __host__ __device__ inline T* end() const;
        __host__ __device__ inline T& front() const;
        __host__ __device__ inline unsigned int getCapacity() const;
        __host__ __device__ inline unsigned int getSize() const;
        __host__ __device__ inline bool isEmpty() const;
        __host__ __device__ inline bool isFull() const;
    private:
        __host__ __device__ static T* mallocStorage(unsigned int capacity, Memory::MallocType allocType);
    public:
        __host__ __device__ StaticVector<T>& operator=(StaticVector<T> const & other);
        __host__ __device__ inline T& operator[](unsigned int index) const;
        __host__ __device__ inline void popBack();
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ void pushBack(T const & t);
        __host__ __device__ inline void resize(unsigned int size);
        __host__ __device__ static inline std::size_t sizeOfStorage(unsigned int capacity);
        __host__ __device__ inline std::byte* storageEnd() const;
        __host__ __device__ void swap(StaticVector<T>& other);
};

template<typename T>
__host__ __device__
StaticVector<T>::StaticVector(unsigned int capacity, Memory::MallocType allocType) :
    flags(Flag::Owning),
    size(0),
    capacity(capacity),
    storage(mallocStorage(capacity, allocType))
{}

template<typename T>
__host__ __device__
StaticVector<T>::StaticVector(unsigned int capacity, std::byte* storage) :
    flags(0),
    size(0),
    capacity(capacity),
    storage(reinterpret_cast<T*>(storage))
{}

template<typename T>
__host__ __device__
StaticVector<T>::StaticVector(T* begin, T* end) :
    flags(0),
    size(0),
    capacity(thrust::distance(begin, end)),
    storage(begin)
{}

template<typename T>
__host__ __device__
StaticVector<T>::~StaticVector()
{
    if(flags & Flag::Owning)
    {
        free(storage);
    }
}

template<typename T>
__host__ __device__
T& StaticVector<T>::at(unsigned int index) const
{
    assert(size > 0);
    assert(index < capacity);

    return storage[index];
}

template<typename T>
__host__ __device__
T& StaticVector<T>::back() const
{
    return at(size - 1);
}

template<typename T>
__host__ __device__
T* StaticVector<T>::begin() const
{
    return storage;
}

template<typename T>
__host__ __device__
void StaticVector<T>::clear()
{
    size = 0;
}

template<typename T>
__host__ __device__
T* StaticVector<T>::end() const
{
    return storage + size;
}

template<typename T>
__host__ __device__
T& StaticVector<T>::front() const
{
    return at(0);
}

template<typename T>
__host__ __device__
unsigned int StaticVector<T>::getCapacity() const
{
    return capacity;
}

template<typename T>
__host__ __device__
unsigned int StaticVector<T>::getSize() const
{
    return size;
}

template<typename T>
__host__ __device__
bool StaticVector<T>::isEmpty() const
{
    return size == 0;
}

template<typename T>
__host__ __device__
bool StaticVector<T>::isFull() const
{
    return size == capacity;
}

template<typename T>
__host__ __device__
T* StaticVector<T>::mallocStorage(unsigned int capacity, Memory::MallocType allocType)
{
    std::byte* storage = Memory::safeMalloc(sizeOfStorage(capacity), allocType);
    return reinterpret_cast<T*>(storage);
}

template<typename T>
__host__ __device__
StaticVector<T>& StaticVector<T>::operator=(StaticVector<T> const & other)
{
    resize(other.size);
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
T& StaticVector<T>::operator[](unsigned int index) const
{
    return at(index);
}

template<typename T>
__host__ __device__
void StaticVector<T>::popBack()
{
    resize(size - 1);
}


template<typename T>
__host__ __device__
void StaticVector<T>::print(bool endLine) const
{
    static_assert(std::is_integral<T>::value);

    printf("[");
    if(size > 0)
    {
        printf("%d", at(0));
        for (unsigned int i = 1; i < size; i += 1)
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
void StaticVector<T>::pushBack(T const & t)
{
    resize(size + 1);
    back() = t;
}

template<typename T>
__host__ __device__
void StaticVector<T>::resize(unsigned int size)
{
    assert(size <= capacity);
    this->size = size;
}

template<typename T>
__host__ __device__
std::size_t StaticVector<T>::sizeOfStorage(unsigned int capacity)
{
    return sizeof(T) * capacity;
}

template<typename T>
__host__ __device__
std::byte* StaticVector<T>::storageEnd() const
{
    return reinterpret_cast<std::byte*>(storage + capacity);
}

template<typename T>
__host__ __device__
void StaticVector<T>::swap(StaticVector<T>& other)
{
    thrust::swap(flags, other.flags);
    thrust::swap(size, other.size);
    thrust::swap(capacity, other.capacity);
    thrust::swap(storage, other.storage);
}