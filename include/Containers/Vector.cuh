#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/swap.h>
#include <Containers/LightVector.cuh>
#include <Utils/Memory.cuh>

template<typename T>
class Vector : public LightVector<T>
{
    // Functions
    public:
        __host__ __device__ Vector(std::size_t capacity, Memory::MallocType mallocType);
        __host__ __device__ ~Vector();
        __host__ __device__ void operator=(Vector<T> const & other);
    private:
        __host__ __device__ static T* mallocStorage(std::size_t capacity, Memory::MallocType mallocType);
};


template<typename T>
__host__ __device__
Vector<T>::Vector(std::size_t capacity, Memory::MallocType mallocType) :
    LightVector<T>(capacity, mallocStorage(capacity, mallocType))
{}

template<typename T>
__host__ __device__
Vector<T>::~Vector()
{
    free(this->storage);
}

template<typename T>
__host__ __device__
void Vector<T>::operator=(Vector<T> const & other)
{
    resize(other.getSize());
    thrust::copy(thrust::seq, other.begin(), other.end(), this->begin());
}

template<typename T>
__host__ __device__
T* Vector<T>::mallocStorage(std::size_t capacity, Memory::MallocType mallocType)
{
    return reinterpret_cast<T*>(Memory::safeMalloc(LightArray<T>::sizeOfStorage(capacity), mallocType));
}
