#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <Extra/Utils/Algorithms.cuh>


template<typename T>
class RuntimeArray
{
    private:
        bool const owning: 1;
    public:
        uint32_t const size: 31;
    private:
        T * const storage;

    public:
        __host__ __device__ RuntimeArray(unsigned int size);
        __host__ __device__ RuntimeArray(unsigned int size, T * const storage);
        __host__ __device__ RuntimeArray(unsigned int size, std::byte * const storage);
        __host__ __device__ ~RuntimeArray();
        __host__ __device__ T & operator[](unsigned int index) const;
        __host__ __device__ bool operator==(RuntimeArray<T> const & other) const;
        __host__ __device__ T & at(unsigned int index) const;
        __host__ __device__ T & front() const;
        __host__ __device__ T & back() const;
        __host__ __device__ T * begin() const;
        __host__ __device__ T * end() const;

        __host__ __device__ static std::size_t sizeofStorage(unsigned int capacity);
};

template<typename T>
__host__ __device__
RuntimeArray<T>::RuntimeArray(unsigned int size) :
    owning(true),
    size(size),
    storage(static_cast<T*>(malloc(sizeof(T) * size)))
{}

template<typename T>
__host__ __device__
RuntimeArray<T>::RuntimeArray(unsigned int size, T * const storage) :
    owning(false),
    size(size),
    storage(storage)
{}

template<typename T>
__host__ __device__
RuntimeArray<T>::RuntimeArray(unsigned int size, std::byte * const storage) :
    RuntimeArray<T>(size, reinterpret_cast<T*>(storage))
{}

template<typename T>
__host__ __device__ RuntimeArray<T>::~RuntimeArray()
{
    if(owning)
    {
        free(storage);
    }
}

template<typename T>
__host__ __device__
T & RuntimeArray<T>::operator[](unsigned int index) const
{
    assert(index < size);
    return storage[index];
}

template<typename T>
__host__ __device__
bool RuntimeArray<T>::operator==(RuntimeArray<T> const & other) const
{
    return size == other.size and Algorithms::equal(begin(), end(), other.begin());

}

template<typename T>
__host__ __device__
T & RuntimeArray<T>::at(unsigned int index) const
{
    return operator[](index);
}

template<typename T>
__host__ __device__
T & RuntimeArray<T>::front() const
{
    return at(0);
}

template<typename T>
__host__ __device__
T & RuntimeArray<T>::back() const
{
    return at(size - 1);
}

template<typename T>
__host__ __device__
T * RuntimeArray<T>::begin() const
{
    return storage;
}

template<typename T>
__host__ __device__
T * RuntimeArray<T>::end() const
{
    return storage + size;
}

template<typename T>
__host__ __device__
std::size_t RuntimeArray<T>::sizeofStorage(unsigned int size)
{
    return sizeof(T) * size;
}
