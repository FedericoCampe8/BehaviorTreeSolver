#pragma once

#include <Containers/LightVector.cuh>

template<typename T>
class Vector : public LightVector<T>
{
    // Functions
    public:
    __host__ __device__ Vector(unsigned int capacity, T* storage);
    __host__ __device__ Vector(unsigned int capacity, Memory::MallocType mallocType);
    __host__ __device__ ~Vector();
    __host__ __device__ Vector<T>& operator=(Vector<T> const & other);
    private:
    __host__ __device__ static T* mallocStorage(unsigned int capacity, Memory::MallocType mallocType);
};

template<typename T>
__host__ __device__
Vector<T>::Vector(unsigned int capacity, T* storage) :
    LightVector<T>(capacity, storage)
{}

template<typename T>
__host__ __device__
Vector<T>::Vector(unsigned int capacity, Memory::MallocType mallocType) :
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
Vector<T>& Vector<T>::operator=(Vector<T> const & other)
{
    resize(other.getSize());
    thrust::copy(thrust::seq, other.begin(), other.end(), this->begin());
    return *this;
}

template<typename T>
__host__ __device__
T* Vector<T>::mallocStorage(unsigned int capacity, Memory::MallocType mallocType)
{
    unsigned int const memorySize = LightArray<T>::sizeOfStorage(capacity);
    std::byte* memory = Memory::safeMalloc(memorySize, mallocType);
    return reinterpret_cast<T*>(memory);
}
