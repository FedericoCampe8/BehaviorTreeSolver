#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cassert>
#include <type_traits>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <Utils/Memory.cuh>

template<typename T>
class RuntimeArray
{

    public:
        uint32_t const size: 31;
    private:
        bool const owning : 1;
        T * const storage;

    public:
        __host__ __device__ RuntimeArray(unsigned int size);
        __host__ __device__ RuntimeArray(T* begin, T* end);
        __host__ __device__ RuntimeArray(unsigned int size, std::byte* storage);
        __host__ __device__ ~RuntimeArray();
        __host__ __device__ T& operator[](unsigned int index) const;
        __host__ __device__ RuntimeArray<T>& operator=(RuntimeArray<T> const & other);
        __host__ __device__ bool operator==(RuntimeArray<T> const & other) const;
        __host__ __device__ T& at(unsigned int index) const;
        __host__ __device__ T* begin() const;
        __host__ __device__ T* end() const;
        __host__ __device__ void print(bool endline = true) const;
        __host__ __device__ std::byte* getStorageEnd(size_t const alignment = 1) const;
        __host__ __device__ void swap(RuntimeArray<T> & other);
        __host__ __device__ static std::size_t sizeofStorage(unsigned int size);
    private:
        __host__ __device__ T* mallocStorage();
        __host__ __device__ RuntimeArray(bool owning, unsigned int size, T* storage);
};

template<typename T>
__host__ __device__
RuntimeArray<T>::RuntimeArray(unsigned int size) :
    size(size),
    owning(true),
    storage(mallocStorage())
{}

template<typename T>
__host__ __device__
RuntimeArray<T>::RuntimeArray(T* begin, T* end) :
    RuntimeArray<T>(static_cast<unsigned int>(thrust::distance(begin,end)), begin)
{}

template<typename T>
__host__ __device__
RuntimeArray<T>::RuntimeArray(unsigned int size, std::byte* storage) :
    size(size),
    owning(false),
    storage(reinterpret_cast<T*>(storage))
{}

template<typename T>
__host__ __device__
RuntimeArray<T>::~RuntimeArray()
{
    if(owning)
    {
        std::free(storage);
    }
}

template<typename T>
__host__ __device__
T& RuntimeArray<T>::operator[](unsigned int index) const
{
    assert(size > 0);
    assert(index < size);
    return storage[index];
}

template<typename T>
__host__ __device__
RuntimeArray<T> & RuntimeArray<T>::operator=(RuntimeArray<T> const & other)
{
    assert(other.size == size);
    thrust::copy(thrust::seq, other.begin(), other.end(), begin());
    return *this;
}

template<typename T>
__host__ __device__
bool RuntimeArray<T>::operator==(RuntimeArray<T> const & other) const
{
    return size == other.size and thrust::equal(begin(), end(), other.begin());
}

template<typename T>
__host__ __device__
std::size_t RuntimeArray<T>::sizeofStorage(unsigned int size)
{
    return sizeof(T) * size;
}

template<typename T>
__host__ __device__
T& RuntimeArray<T>::at(unsigned int index) const
{
    return operator[](index);
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
    return storage + size;
}

template<typename T>
__host__ __device__
void RuntimeArray<T>::print(bool endline) const
{
    static_assert(std::is_integral<T>::value);

    printf("[");
    if(size > 0)
    {
        printf("%d", storage[0]);
        for (uint i = 1; i < size; i += 1)
        {
            printf(",%d", storage[i]);
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
T* RuntimeArray<T>::mallocStorage()
{
    std::byte* storage = Memory::safeMalloc(sizeofStorage(size));
    return reinterpret_cast<T*>(storage);
}

template<typename T>
__host__ __device__
std::byte* RuntimeArray<T>::getStorageEnd(size_t const alignment) const
{
    return Memory::align(alignment,reinterpret_cast<std::byte*>(storage + size));
}

template<typename T>
__host__ __device__
void RuntimeArray<T>::swap(RuntimeArray<T>& other)
{
    RuntimeArray<T> tmp(owning, size, storage);
    new (this) RuntimeArray<T>(other.owning, other.size, other.storage);
    new (&other) RuntimeArray<T>(tmp.owning, tmp.size, tmp.storage);
}

template<typename T>
__host__ __device__
RuntimeArray<T>::RuntimeArray(bool owning, unsigned int size, T* storage) :
    size(size),
    owning(owning),
    storage(storage)
{}

