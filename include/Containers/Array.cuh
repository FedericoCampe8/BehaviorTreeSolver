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
class Array : public LightArray<T>
{
    // Functions
    public:
        __host__ __device__ Array(unsigned int capacity, Memory::MallocType mallocType);
        __host__ __device__ ~Array();
        __host__ __device__ void operator=(LightArray<T> const & other);

    private:
        __host__ __device__ static T* mallocStorage(std::size_t capacity, Memory::MallocType mallocType);
};

template<typename T>
__host__ __device__
Array<T>::Array(unsigned int capacity, Memory::MallocType mallocType) :
    LightArray<T>(capacity, mallocStorage(capacity, mallocType))
{}

template<typename T>
__host__ __device__
Array<T>::~Array()
{
    free(this->storage);
}

template<typename T>
__host__ __device__
void Array<T>::operator=(LightArray<T> const & other)
{
    assert(this->capacity == other.getCapacity());
    thrust::copy(thrust::seq, other.begin(), other.end(), this->begin());
}

template<typename T>
__host__ __device__
T* Array<T>::mallocStorage(std::size_t capacity, Memory::MallocType mallocType)
{
    unsigned int memorySize = LightArray<T>::sizeOfStorage(capacity);
    std::byte* memory = Memory::safeMalloc(memorySize, mallocType);
    return reinterpret_cast<T*>(memory);
}
