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
class StaticVector
{
    public:
        uint32_t const capacity;
    private:
        bool const owning : 1;
        uint32_t size : 31;
        T * const storage;

    public:
        __host__ __device__ StaticVector(unsigned int capacity);
        __host__ __device__ StaticVector(unsigned int capacity, std::byte* storage);
        __host__ __device__ ~StaticVector();
        __host__ __device__ T& operator[](unsigned int index) const;
        __host__ __device__ StaticVector<T>& operator=(StaticVector<T> const & other);
        __host__ __device__ bool operator==(StaticVector<T> const & other) const;
        __host__ __device__ unsigned int getSize() const;
        __host__ __device__ bool isEmpty() const;
        __host__ __device__ void resize(unsigned int size);
        __host__ __device__ T& at(unsigned int index) const;
        __host__ __device__ T& front() const;
        __host__ __device__ T& back() const;
        __host__ __device__ T* begin() const;
        __host__ __device__ T* end() const;
        template<typename ... Args>  __host__ __device__ void emplaceBack(Args ... args);
        __host__ __device__ void pushBack(T const & t);
        __host__ __device__ void popBack();
        __host__ __device__ void clear();
        __host__ __device__ void remove(T const & t);
        __host__ __device__ void print(bool endline = true) const;
        __host__ __device__ std::byte* getStorageEnd() const;
        __host__ __device__ static std::size_t sizeofStorage(unsigned int capacity);
    private:
        __host__ __device__ T* mallocStorage();
};

template<typename T>
__host__ __device__
StaticVector<T>::StaticVector(unsigned int capacity) :
    capacity(capacity),
    owning(true),
    size(0),
    storage(mallocStorage())
{}

template<typename T>
__host__ __device__
StaticVector<T>::StaticVector(unsigned int capacity, std::byte* storage) :
    capacity(capacity),
    owning(false),
    size(0),
    storage(reinterpret_cast<T*>(storage))
{}

template<typename T>
__host__ __device__
StaticVector<T>::~StaticVector()
{
    if(owning)
    {
        std::free(storage);
    }
}

template<typename T>
__host__ __device__
T& StaticVector<T>::operator[](unsigned int index) const
{
    assert(size > 0);
    assert(index < capacity);
    return storage[index];
}

template<typename T>
__host__ __device__
StaticVector<T>& StaticVector<T>::operator=(StaticVector<T> const & other)
{
    assert(other.size <= capacity);
    size = other.size;
    thrust::copy(thrust::seq, other.begin(), other.end(), begin());
    return *this;
}

template<typename T>
__host__ __device__
bool StaticVector<T>::operator==(StaticVector<T> const & other) const
{
    return size == other.size and thrust::equal(begin(), end(), other.begin());
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
std::size_t StaticVector<T>::sizeofStorage(unsigned int capacity)
{
    return sizeof(T) * capacity;
}

template<typename T>
__host__ __device__
void StaticVector<T>::resize(unsigned int size)
{
    this->size = size;
}

template<typename T>
__host__ __device__
T& StaticVector<T>::at(unsigned int index) const
{
    return operator[](index);
}

template<typename T>
__host__ __device__
T& StaticVector<T>::front() const
{
    return operator[](0);
}

template<typename T>
__host__ __device__
T& StaticVector<T>::back() const
{
    return operator[](size -1);
}

template<typename T>
__host__ __device__
T* StaticVector<T>::begin() const
{
    return storage;
}

template<typename T>
__host__ __device__
T* StaticVector<T>::end() const
{
    return storage + size;
}

template<typename T>
template<typename ... Args>
__host__ __device__
void StaticVector<T>::emplaceBack(Args ... args)
{
    assert(size < capacity);
    new (&storage[size]) T(args ...);
    size += 1;
}

template<typename T>
__host__ __device__
void StaticVector<T>::clear()
{
    size = 0;
}

template<typename T>
__host__ __device__
void StaticVector<T>::popBack()
{
    assert(size > 0);
    size -= 1;
}

template<typename T>
__host__ __device__
void StaticVector<T>::print(bool endline) const
{
    static_assert(std::is_integral<T>::value);

    printf("[");
    if(size > 0)
    {
        printf("%d",storage[0]);
        for (uint i = 1; i < size; i += 1)
        {
            printf(",%d", storage[i]);
        }
        for (uint i = size; i < capacity; i += 1)
        {
            printf(",_");
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
void StaticVector<T>::pushBack(T const & t)
{
    assert(size < capacity);
    storage[size] = t;
    size += 1;
}

template<typename T>
__host__ __device__
T* StaticVector<T>::mallocStorage()
{
    std::byte* storage = Memory::safeMalloc(sizeofStorage(capacity));
    return static_cast<T*>(storage);
}

template<typename T>
__host__ __device__
std::byte* StaticVector<T>::getStorageEnd() const
{
    return reinterpret_cast<std::byte*>(storage + capacity);
}

template<typename T>
__host__ __device__
void StaticVector<T>::remove(T const & t)
{
    T* end = &storage[size];
    end = thrust::remove(thrust::seq, storage, end, t);
    size = thrust::distance(storage, end);
}
