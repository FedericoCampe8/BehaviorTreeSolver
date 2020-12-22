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
class Vector : public LightArray
{
    // Members
    private:
        std::size_t size;

    // Functions
    public:
        __host__ __device__ Vector(std::size_t capacity, Memory::MallocType mallocType);
        __host__ __device__ ~Vector();
        __host__ __device__ inline void clear();
        template<typename ... Args>
        __host__ __device__ void emplaceBack(Args ... args);
        __host__ __device__ inline std::size_t getSize() const;
        __host__ __device__ inline bool isEmpty() const;
        __host__ __device__ inline bool isFull() const;
        __host__ __device__ void operator=(Vector<T> const * other);
        __host__ __device__ inline T* operator[](std::size_t index) const;
        __host__ __device__ inline void popBack();
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ void pushBack(T const * t);
        __host__ __device__ inline void resize(std::size_t size);
        __host__ __device__ static void swap(Vector<T>* v0, Vector<T>* v1);
    private:
        __host__ __device__ static T* mallocStorage(std::size_t capacity, Memory::MallocType mallocType);
};


template<typename T>
__host__ __device__
Vector<T>::Vector(std::size_t capacity, Memory::MallocType mallocType) :
    LightArray(capacity, mallocStorage(capacity, mallocType)),
    size(0)
{}

template<typename T>
__host__ __device__
Vector<T>::~Vector()
{
    free(storage);
}

template<typename T>
__host__ __device__
void Vector<T>::clear()
{
    size = 0;
}

template<typename T>
template<typename ... Args>
void Vector<T>::emplaceBack(Args ... args)
{
    resize(size + 1);
    new (back()) T(args ...);
}

template<typename T>
__host__ __device__
std::size_t Vector<T>::getSize() const
{
    return size;
}

template<typename T>
__host__ __device__
bool Vector<T>::isEmpty() const
{
    return size == 0;
}

template<typename T>
__host__ __device__
bool Vector<T>::isFull() const
{
    return size == capacity;
}

template<typename T>
__host__ __device__
void Vector<T>::operator=(Vector<T> const * other)
{
    resize(other->size);
    thrust::copy(thrust::seq, other->begin(), other->end(), begin());
}

template<typename T>
__host__ __device__
T* Vector<T>::operator[](std::size_t index) const
{
    assert(index < size);
    LightArray::operator[](index);
}


template<typename T>
__host__ __device__
void Vector<T>::popBack()
{
    resize(size - 1);
}

template<typename T>
__host__ __device__
void Vector<T>::print(bool endLine) const
{
   LightArray::print(0, size, endLine);
}

template<typename T>
__host__ __device__
void Vector<T>::pushBack(T const * t)
{
    resize(size + 1);
    *back() = *t;
}

template<typename T>
__host__ __device__
void Vector<T>::resize(std::size_t size)
{
    assert(size <= capacity);
    this->size = size;
}

template<typename T>
__host__ __device__
void Vector<T>::swap(Vector<T>* v0, Vector<T>* v1)
{
    LightArray<T>::swap(v0, v1);
    thrust::swap(v0->size, v1->size);
}

template<typename T>
__host__ __device__
T* Vector<T>::mallocStorage(std::size_t capacity, Memory::MallocType mallocType)
{
    return reinterpret_cast<T*>(Memory::safeMalloc(LightArray<T>::sizeOfStorage(capacity), mallocType));
}
