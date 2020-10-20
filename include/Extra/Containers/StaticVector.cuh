#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <Extra/Utils/Algorithms.cuh>

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
        __host__ __device__ StaticVector(unsigned int capacity, T * const storage);
        __host__ __device__ StaticVector(unsigned int capacity, std::byte * const storage);
        __host__ __device__ ~StaticVector();
        __host__ __device__ T & operator[](unsigned int index) const;
        __host__ __device__ StaticVector<T> & operator=(StaticVector<T> const & other);
        __host__ __device__ bool operator==(StaticVector<T> const & other) const;
        __host__ __device__ unsigned int getSize() const;
        __host__ __device__ void resize(unsigned int size);
        __host__ __device__ T & at(unsigned int index) const;
        __host__ __device__ T & front() const;
        __host__ __device__ T & back() const;
        __host__ __device__ T * begin() const;
        __host__ __device__ T * end() const;
        template<typename ... Args>  __host__ __device__ void emplaceBack(Args ... args);
        __host__ __device__ void pushBack(T const & t);
        __host__ __device__ void popBack();
        __host__ __device__ void clear();
        __host__ __device__ void print(bool newline = false) const;

        __host__ __device__ static std::size_t sizeofStorage(unsigned int capacity);
};

template<typename T>
__host__ __device__
StaticVector<T>::StaticVector(unsigned int capacity) :
    owning(true),
    capacity(capacity),
    size(0),
    storage(static_cast<T*>(std::malloc(sizeof(T) * capacity)))
{}

template<typename T>
__host__ __device__
StaticVector<T>::StaticVector(unsigned int capacity, T * const storage) :
    owning(false),
    capacity(capacity),
    size(0),
    storage(storage)
{}

template<typename T>
__host__ __device__
StaticVector<T>::StaticVector(unsigned int capacity, std::byte * const storage) :
    StaticVector<T>(capacity, reinterpret_cast<T*>(storage))
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
T &StaticVector<T>::operator[](unsigned int index) const
{
    assert(size > 0);
    assert(index < capacity);
    return storage[index];
}

template<typename T>
__host__ __device__
StaticVector<T> & StaticVector<T>::operator=(StaticVector<T> const & other)
{
    assert(other.size <= capacity);
    size = other.size;
    Algorithms::copy( other.begin(), other.end(), begin());
    return *this;
}

template<typename T>
__host__ __device__
bool StaticVector<T>::operator==(StaticVector<T> const & other) const
{
    return size == other.size and Algorithms::equal(begin(), end(), other.begin());
}

template<typename T>
__host__ __device__
unsigned int StaticVector<T>::getSize() const
{
    return size;
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
T & StaticVector<T>::at(unsigned int index) const
{
    return operator[](index);
}

template<typename T>
__host__ __device__
T & StaticVector<T>::front() const
{
    return at(0);
}

template<typename T>
__host__ __device__
T & StaticVector<T>::back() const
{
    return at(size - 1);
}

template<typename T>
__host__ __device__
T * StaticVector<T>::begin() const
{
    return storage;
}

template<typename T>
__host__ __device__
T * StaticVector<T>::end() const
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
void StaticVector<T>::print(bool newline) const
{
    printf("[");
    if(size > 0)
    {
        printf("%d",storage[0]);
        for (uint i = 1; i < size; i += 1)
        {
            printf(",%d",storage[i]);
        }
    }
    printf("]");
    if(newline)
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
