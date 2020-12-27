#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/swap.h>
#include <Containers/LightArray.cuh>
#include <Utils/Memory.cuh>

template<typename T>
class LightVector : public LightArray<T>
{
    // Members
    protected:
        std::size_t size;

    // Functions
    public:
        __host__ __device__ LightVector(std::size_t capacity, T* storage);
        __host__ __device__ ~LightVector();
        __host__ __device__ inline T* at(std::size_t index) const;
        __host__ __device__ inline T* back() const;
        __host__ __device__ inline void clear();
        __host__ __device__ inline T* end() const;
        __host__ __device__ inline T* front() const;
        __host__ __device__ inline std::size_t getSize() const;
        __host__ __device__ inline bool isEmpty() const;
        __host__ __device__ inline bool isFull() const;
        __host__ __device__ inline void incrementSize(std::size_t increment = 1);
        __host__ __device__ inline std::size_t indexOf(T const * t) const;
        __host__ __device__ void operator=(LightVector<T> const & other);
        __host__ __device__ inline T* operator[](std::size_t index) const;
        __host__ __device__ inline void popBack();
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ inline void resize(std::size_t size);
        __host__ __device__ static void swap(LightVector<T>* v0, LightVector<T>* v1);
};


template<typename T>
__host__ __device__
LightVector<T>::LightVector(std::size_t capacity, T* storage) :
    LightArray<T>(capacity, storage),
    size(0)
{}

template<typename T>
__host__ __device__
LightVector<T>::~LightVector()
{}

template<typename T>
__host__ __device__
T* LightVector<T>::at(std::size_t index) const
{
    assert(index < size);
    return LightArray<T>::at(index);
}

template<typename T>
__host__ __device__
T* LightVector<T>::back() const
{
    assert(size > 0);
    return at(size - 1);
}

template<typename T>
__host__ __device__
void LightVector<T>::clear()
{
    size = 0;
}
template<typename T>
__host__ __device__
T* LightVector<T>::end() const
{
    return this->storage + size;
}

template<typename T>
__host__ __device__
T* LightVector<T>::front() const
{
    assert(size > 0);
    return this->at(0);
}

template<typename T>
__host__ __device__
std::size_t LightVector<T>::getSize() const
{
    return size;
}

template<typename T>
__host__ __device__
bool LightVector<T>::isEmpty() const
{
    return size == 0;
}

template<typename T>
__host__ __device__
bool LightVector<T>::isFull() const
{
    return size == this->capacity;
}

template<typename T>
__host__ __device__
void LightVector<T>::incrementSize(std::size_t increment)
{
    resize(size + increment);
}

template<typename T>
__host__ __device__
std::size_t LightVector<T>::indexOf(T const * t) const
{
    T const * const b = this->begin();

    assert(b <= t);
    assert(t < end());

    return thrust::distance(b, t);
}

template<typename T>
__host__ __device__
void LightVector<T>::operator=(LightVector<T> const & other)
{
    LightArray<T>::operator=(other);
    size = other.size;
}

template<typename T>
__host__ __device__
T* LightVector<T>::operator[](std::size_t index) const
{
    return at(index);
}

template<typename T>
__host__ __device__
void LightVector<T>::popBack()
{
    resize(size - 1);
}

template<typename T>
__host__ __device__
void LightVector<T>::print(bool endLine) const
{
    LightArray<T>::print(0, size, endLine);
}

template<typename T>
__host__ __device__
void LightVector<T>::resize(std::size_t size)
{
    assert(size <= this->capacity);
    this->size = size;
}

template<typename T>
__host__ __device__
void LightVector<T>::swap(LightVector<T>* v0, LightVector<T>* v1)
{
    LightArray<T>::swap(v0, v1);
    thrust::swap(v0->size, v1->size);
}
