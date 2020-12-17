#pragma once

#include <cstdint>
#include <utility>

#include <thrust/distance.h>
#include <thrust/sequence.h>

#include <Containers/RuntimeArray.cuh>
#include <Containers/StaticVector.cuh>
#include <Utils/Memory.cuh>

template<typename T>
class StaticSet
{
    private:
        RuntimeArray<T> elements;
        StaticVector<uint32_t> invalids;

    public:
        __host__ __device__ StaticSet(unsigned int capacity, Memory::MallocType mallocType);
        __host__ __device__ StaticSet(StaticSet<T>&& other);
        __host__ __device__ inline T& at(unsigned int index) const;
        __host__ __device__ void clear();
        __host__ __device__ inline void erase(T const * t);
        __host__ __device__ inline unsigned int getCapacity() const;
        __host__ __device__ T* insert(T const & t);
        __host__ __device__ inline bool isEmpty() const;
        __host__ __device__ inline bool isFull() const;
        __host__ __device__ T& operator=(StaticSet<T>&& other) const;
        __host__ __device__ inline T& operator[](unsigned int index) const;
};

template<typename T>
__host__ __device__
StaticSet<T>::StaticSet(unsigned int capacity, Memory::MallocType mallocType) :
    elements(capacity, mallocType),
    invalids(capacity, mallocType)
{
    invalids.resize(capacity);
    thrust::sequence(thrust::seq, invalids.begin(), invalids.end());
}

template<typename T>
__host__ __device__
StaticSet<T>::StaticSet(StaticSet<T>&& other) :
    elements(std::move(other.elements)),
    invalids(std::move(other.invalids))
{}


template<typename T>
__host__ __device__
T& StaticSet<T>::at(unsigned int index) const
{
    return elements.at(index);
}

template<typename T>
__host__ __device__
void StaticSet<T>::clear()
{
    invalids.resize(invalids.getCapacity());
    thrust::sequence(thrust::seq, invalids.begin(), invalids.end());
}

template<typename T>
__host__ __device__
void StaticSet<T>::erase(T const * t)
{
    assert(elements.begin() <= t);
    assert(t < elements.end());

    unsigned int const elementIdx = thrust::distance(elements.begin(), t);
    invalids.pushBack(elementIdx);
}

template<typename T>
__host__ __device__
unsigned int StaticSet<T>::getCapacity() const
{
    return elements.getCapacity();
}

template<typename T>
__host__ __device__
T* StaticSet<T>::insert(T const & t)
{
    assert(not isFull());

    unsigned int const elementIdx = invalids.back();
    invalids.popBack();
    T& element = element.at(elementIdx);
    element = t;
    return &element;
}

template<typename T>
__host__ __device__
bool StaticSet<T>::isEmpty() const
{
    return invalids.isFull();
}

template<typename T>
__host__ __device__
bool StaticSet<T>::isFull() const
{
    return invalids.isEmpty();
}

template<typename T>
__host__ __device__
T& StaticSet<T>::operator[](unsigned int index) const
{
    return elements[index];
}