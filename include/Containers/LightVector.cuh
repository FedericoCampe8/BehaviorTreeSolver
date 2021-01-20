#pragma once

#include <Containers/LightArray.cuh>

template<typename T>
class LightVector : public LightArray<T>
{
    // Members
    protected:
    unsigned int size;

    // Functions
    public:
    __host__ __device__ LightVector(unsigned int capacity, T* storage);
    __host__ __device__ ~LightVector();
    __host__ __device__ inline T* at(unsigned int index) const;
    __host__ __device__ inline T* back() const;
    __host__ __device__ inline void clear();
    __host__ __device__ inline T* end() const;
    __host__ __device__ inline T* front() const;
    __host__ __device__ inline unsigned int getSize() const;
    __host__ __device__ inline bool isEmpty() const;
    __host__ __device__ inline bool isFull() const;
    __host__ __device__ inline unsigned int indexOf(T const * t) const;
    __host__ __device__ LightVector<T>& operator=(LightVector<T> const & other);
    __host__ __device__ inline T* operator[](unsigned int index) const;
    __host__ __device__ inline void popBack();
    __host__ __device__ void print(bool endLine = true) const;
    __host__ __device__ void pushBack(T const * t);
    __host__ __device__ inline void resize(unsigned int size);
    __host__ __device__ inline static void swap(LightVector<T>& v0, LightVector<T>& v1);
};

template<typename T>
__host__ __device__
LightVector<T>::LightVector(unsigned int capacity, T* storage) :
    LightArray<T>(capacity, storage),
    size(0)
{}

template<typename T>
__host__ __device__
LightVector<T>::~LightVector()
{}

template<typename T>
__host__ __device__
T* LightVector<T>::at(unsigned int index) const
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
    return at(0);
}

template<typename T>
__host__ __device__
unsigned int LightVector<T>::getSize() const
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
unsigned int LightVector<T>::indexOf(T const * t) const
{
    T const* const b = this->begin();
    assert(b <= t);
    assert(t < end());
    return static_cast<unsigned int>(thrust::distance(b, t));
}

template<typename T>
__host__ __device__
LightVector<T>& LightVector<T>::operator=(LightVector<T> const & other)
{
    LightArray<T>::operator=(other);
    size = other.size;
    return *this;
}

template<typename T>
__host__ __device__
T* LightVector<T>::operator[](unsigned int index) const
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
void LightVector<T>::pushBack(T const * t)
{
    resize(size + 1);
    *back() = *t;
}

template<typename T>
__host__ __device__
void LightVector<T>::resize(unsigned int size)
{
    assert(size <= this->capacity);
    this->size = size;
}

template<typename T>
__host__ __device__
void LightVector<T>::swap(LightVector<T>& v0, LightVector<T>& v1)
{
    LightArray<T>::swap(v0, v1);
    thrust::swap(v0.size, v1.size);
}
