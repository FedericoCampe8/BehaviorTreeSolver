#pragma once

#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include <Containers/RuntimeArray.cuh>
#include <Containers/StaticVector.cuh>

template<typename T>
class StaticSet
{
    private:
        RuntimeArray<T>* const elements;
        StaticVector<unsigned int>* const invalids;

    public:
        __host__ __device__ StaticSet(RuntimeArray<T>* elements, StaticVector<unsigned int>* invalids);
        __host__ __device__ inline T& at(unsigned int index) const;
        __host__ __device__ T* add(T const & t);
        __host__ __device__ T* begin() const;
        __host__ __device__ T* end() const;
        __host__ __device__ inline unsigned int getCapacity() const;
        __host__ __device__ inline bool isEmpty() const;
        __host__ __device__ inline bool isFull() const;
        __host__ __device__ inline T& operator[](unsigned int index) const;
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ void remove(T* t);
        __host__ __device__ void reset();
};

template<typename T>
__host__ __device__
StaticSet<T>::StaticSet(RuntimeArray<T>* elements, StaticVector<unsigned int>* invalids) :
    elements(elements),
    invalids(invalids)
{
    assert(elements->getCapacity() == invalids->getCapacity());

    invalids->resize(invalids->getCapacity());
    thrust::sequence(
#ifdef __CUDA_ARCH__
        thrust::device,
#else
        thrust::host,
#endif
        invalids->begin(), invalids->end());
}

template<typename T>
__host__ __device__
T& StaticSet<T>::at(unsigned int index) const
{
    return elements->at(index);
}

template<typename T>
__host__ __device__
T* StaticSet<T>::add(T const & t)
{
    assert(not isFull());

    unsigned int elementIdx = invalids->back();
    invalids->popBack();
    T& element = elements->at(elementIdx);
    element = t;
    return &element;
}

template<typename T>
__host__ __device__
T* StaticSet<T>::begin() const
{
    return elements->begin();
}

template<typename T>
__host__ __device__
T* StaticSet<T>::end() const
{
    return elements->end();
}

template<typename T>
__host__ __device__
unsigned int StaticSet<T>::getCapacity() const
{
    return elements->getCapacity();
}

template<typename T>
__host__ __device__
bool StaticSet<T>::isEmpty() const
{
    return invalids->isFull();
}

template<typename T>
__host__ __device__
bool StaticSet<T>::isFull() const
{
    return invalids->isEmpty();
}

template<typename T>
__host__ __device__
T& StaticSet<T>::operator[](unsigned int index) const
{
    return elements[index];
}

template<typename T>
__host__ __device__
void StaticSet<T>::print(bool endLine) const
{
    elements->print(endLine);
}

template<typename T>
__host__ __device__
void StaticSet<T>::remove(T* t)
{
    assert(begin() <= t);
    assert(t < end());

    unsigned int elementIdx = thrust::distance(begin(), t);
    invalids->pushBack(elementIdx);
}

template<typename T>
__host__ __device__
void StaticSet<T>::reset()
{
    invalids->resize(invalids->getCapacity());
    thrust::sequence(
#ifdef __CUDA_ARCH__
        thrust::device,
#else
        thrust::host,
#endif
        invalids->begin(), invalids->end());
}