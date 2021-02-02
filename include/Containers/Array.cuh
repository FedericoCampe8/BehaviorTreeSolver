#pragma once

#include <Containers/LightArray.cuh>

template<typename T>
class Array : public LightArray<T>
{
    // Functions
    public:
    __host__ __device__ Array(unsigned int capacity, T* storage);
    __host__ __device__ Array(unsigned int capacity, Memory::MallocType mallocType);
    __host__ __device__ ~Array();
    __host__ __device__ Array<T>& operator=(Array<T> const & other);
    private:
    __host__ __device__ static T* mallocStorage(unsigned int capacity, Memory::MallocType mallocType);
};

template<typename T>
__host__ __device__
Array<T>::Array(unsigned int capacity, T* storage):
    LightArray<T>(capacity, storage)
{}

template<typename T>
__host__ __device__
Array<T>::Array(unsigned int capacity, Memory::MallocType mallocType) :
    Array<T>(capacity, reinterpret_cast<T*>(mallocStorage(capacity, mallocType)))
{}

template<typename T>
__host__ __device__
Array<T>::~Array()
{
    free(this->storage);
}

template<typename T>
__host__ __device__
Array<T>& Array<T>::operator=(Array<T> const & other)
{
    assert(this->capacity == other.capacity);
    thrust::copy(thrust::seq, other.begin(), other.end(), this->begin());
    return *this;
}

template<typename T>
__host__ __device__
T* Array<T>::mallocStorage(unsigned int capacity, Memory::MallocType mallocType)
{
    unsigned int const storageSize = Array<T>::sizeOfStorage(capacity);
    return reinterpret_cast<T*>(Memory::safeMalloc(storageSize, mallocType));
}