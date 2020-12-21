#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/swap.h>
#include <Utils/Memory.cuh>

template<typename T>
class Vector
{
    private:
        enum Flags {Owning = 1};

    private:
        uint32_t flags;
        uint32_t size;
        uint32_t capacity;
        std::byte padding[4]; // 64bit aligned
        T * storage;

    public:
        // Storage
        __host__ __device__ static std::byte* mallocStorage(unsigned int capacity, Memory::MallocType mallocType);
        __host__ __device__ static std::size_t sizeOfStorage(unsigned int capacity);
        __host__ __device__ std::byte* storageEnd() const;

        // Vector
        __host__ __device__ Vector(unsigned int capacity, Memory::MallocType mallocType);
        __host__ __device__ Vector(unsigned int capacity, std::byte* storage);
        __host__ __device__ Vector(T* begin, T* end);
        __host__ __device__ ~Vector();
        __host__ __device__ inline T& at(unsigned int index) const;
        __host__ __device__ inline T& back() const;
        __host__ __device__ inline T* begin() const;
        __host__ __device__ inline void clear();
        __host__ __device__ inline T* end() const;
        __host__ __device__ inline T& front() const;
        __host__ __device__ inline unsigned int getCapacity() const;
        __host__ __device__ inline unsigned int getSize() const;
        __host__ __device__ inline unsigned int indexOf(T const * t) const;
        __host__ __device__ inline bool isEmpty() const;
        __host__ __device__ inline bool isFull() const;
        __host__ __device__ Vector<T>& operator=(Vector<T> const & other);
        __host__ __device__ inline T& operator[](unsigned int index) const;
        __host__ __device__ inline void popBack();
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ void pushBack(T const & t);
        __host__ __device__ inline void resize(unsigned int size);
        __host__ __device__ static void swap(Vector<T>& v0, Vector<T>& v1);
};

template<typename T>
__host__ __device__
std::byte* Vector<T>::mallocStorage(unsigned int capacity, Memory::MallocType mallocType)
{
    return  Memory::safeMalloc(sizeOfStorage(capacity), mallocType);
}

template<typename T>
__host__ __device__
std::size_t Vector<T>::sizeOfStorage(unsigned int capacity)
{
    return sizeof(T) * capacity;
}

template<typename T>
__host__ __device__
std::byte* Vector<T>::storageEnd() const
{
    return reinterpret_cast<std::byte*>(storage + capacity);
}

template<typename T>
__host__ __device__
Vector<T>::Vector(unsigned int capacity, Memory::MallocType mallocType) :
    flags(Flags::Owning),
    size(0),
    capacity(capacity),
    storage(reinterpret_cast<T*>(mallocStorage(capacity,mallocType)))
{}

template<typename T>
__host__ __device__
Vector<T>::Vector(unsigned int capacity, std::byte* storage) :
    flags(0),
    size(0),
    capacity(capacity),
    storage(reinterpret_cast<T*>(storage))
{}

template<typename T>
__host__ __device__
Vector<T>::Vector(T* begin, T* end) :
    flags(0),
    size(0),
    capacity(thrust::distance(begin, end)),
    storage(end)
{}

template<typename T>
__host__ __device__
Vector<T>::~Vector()
{
    if(flags & Flags::Owning)
    {
        free(storage);
    }
}

template<typename T>
__host__ __device__
T& Vector<T>::at(unsigned int index) const
{
    assert(index < size);
    return storage[index];
}

template<typename T>
__host__ __device__
T& Vector<T>::back() const
{
    return at(size - 1);
}

template<typename T>
__host__ __device__
T* Vector<T>::begin() const
{
    return storage;
}

template<typename T>
__host__ __device__
void Vector<T>::clear()
{
    size = 0;
}

template<typename T>
__host__ __device__
T* Vector<T>::end() const
{
    return storage + size;
}

template<typename T>
__host__ __device__
T& Vector<T>::front() const
{
    return at(0);
}

template<typename T>
__host__ __device__
unsigned int Vector<T>::getCapacity() const
{
    return capacity;
}

template<typename T>
__host__ __device__
unsigned int Vector<T>::getSize() const
{
    return size;
}

template<typename T>
__host__ __device__
unsigned int Vector<T>::indexOf(T const * t) const
{
    assert(begin() <= t);
    assert(t < end());

    return thrust::distance(begin(), t);
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
Vector<T>& Vector<T>::operator=(Vector<T> const & other)
{
    resize(other.size);
    thrust::copy(thrust::seq, other.begin(), other.end(), begin());
    return *this;
}

template<typename T>
__host__ __device__
T& Vector<T>::operator[](unsigned int index) const
{
    return at(index);
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
    static_assert(std::is_integral<T>::value);

    printf("[");
    if(size > 0)
    {
        printf("%d", at(0));
        for (unsigned int i = 1; i < size; i += 1)
        {
            printf(",%d", at(i));
        }
    }
    printf("]");
    if (endLine)
    {
        printf("\n");
    }
}

template<typename T>
__host__ __device__
void Vector<T>::pushBack(T const & t)
{
    resize(size + 1);
    back() = t;
}

template<typename T>
__host__ __device__
void Vector<T>::resize(unsigned int size)
{
    assert(size <= capacity);
    this->size = size;
}

template<typename T>
__host__ __device__
void Vector<T>::swap(Vector<T>& v0, Vector<T>& v1)
{
    thrust::swap(v0.flags, v1.flags);
    thrust::swap(v0.size, v1.size);
    thrust::swap(v0.capacity, v1.capacity);
    thrust::swap(v0.storage, v1.storage);
}