#pragma once

#include <cstddef>

#include <Extra/Utils.hh>

namespace Extra::Containers
{
    template<typename T>
    class RestrainedVector
    {
        private:
            bool const owning: 1;
        public:
            uint const capacity: 31;
        private:
            uint size;
            T * const storage;

        public:
            __host__ __device__ RestrainedVector(uint capacity);
            __host__ __device__ RestrainedVector(uint capacity, T * const storage);
            __host__ __device__ RestrainedVector(uint capacity, std::byte * const storage);
            __host__ __device__ ~RestrainedVector();
            __host__ __device__ T & operator[](uint index) const;
            __host__ __device__ RestrainedVector<T> & operator=(RestrainedVector<T> const & other);
            __host__ __device__ bool operator==(RestrainedVector<T> const & other) const;
            __host__ __device__ uint getSize() const;
            __host__ __device__ void resize(uint size);
            __host__ __device__ T & at(uint index) const;
            __host__ __device__ T & front() const;
            __host__ __device__ T & back() const;
            __host__ __device__ T * begin() const;
            __host__ __device__ T * end() const;
            template<typename ... Args>  __host__ __device__ void emplaceBack(Args ... args);
            __host__ __device__ void popBack();
            __host__ __device__ void clear();
    };

    template<typename T>
    __host__ __device__
    RestrainedVector<T>::RestrainedVector(uint capacity) :
        owning(true),
        capacity(capacity),
        size(0),
        storage(static_cast<T*>(Extra::Utils::Memory::malloc(sizeof(T) * capacity)))
    {}

    template<typename T>
    __host__ __device__
    RestrainedVector<T>::RestrainedVector(uint capacity, T * const storage) :
        owning(false),
        capacity(capacity),
        size(0),
        storage(storage)
    {}

    template<typename T>
    __host__ __device__
    RestrainedVector<T>::RestrainedVector(uint capacity, std::byte * const storage) :
        RestrainedVector<T>(capacity, reinterpret_cast<T*>(storage))
    {}

    template<typename T>
    __host__ __device__ RestrainedVector<T>::~RestrainedVector()
    {
        if(owning)
        {
            Extra::Utils::Memory::free(storage);
        }
    }

    template<typename T>
    __host__ __device__
    T &RestrainedVector<T>::operator[](uint index) const
    {
        Extra::Utils::Platform::assert(size > 0);
        Extra::Utils::Platform::assert(index < capacity);
        return storage[index];
    }

    template<typename T>
    __host__ __device__
    RestrainedVector<T> & RestrainedVector<T>::operator=(RestrainedVector<T> const & other)
    {
        Extra::Utils::Platform::assert(other.size <= capacity);
        size = other.size;
        Extra::Algorithms::copy(other.begin(), other.end(), begin());
        return *this;
    }

    template<typename T>
    __host__ __device__
    bool RestrainedVector<T>::operator==(RestrainedVector<T> const & other) const
    {
        return size == other.size and Extra::Algorithms::equal(begin(), end(), other.begin());
    }

    template<typename T>
    __host__ __device__
    uint RestrainedVector<T>::getSize() const
    {
        return size;
    }

    template<typename T>
    __host__ __device__
    void RestrainedVector<T>::resize(uint size)
    {
        this->size = size;
    }

    template<typename T>
    __host__ __device__
    T & RestrainedVector<T>::at(uint index) const
    {
        return operator[](index);
    }

    template<typename T>
    __host__ __device__
    T & RestrainedVector<T>::front() const
    {
        return at(0);
    }

    template<typename T>
    __host__ __device__
    T & RestrainedVector<T>::back() const
    {
        return at(size - 1);
    }

    template<typename T>
    __host__ __device__
    T * RestrainedVector<T>::begin() const
    {
        return storage;
    }

    template<typename T>
    __host__ __device__
    T * RestrainedVector<T>::end() const
    {
        return storage + size;
    }

    template<typename T>
    template<typename ... Args>
    __host__ __device__
    void RestrainedVector<T>::emplaceBack(Args ... args)
    {
        Extra::Utils::Platform::assert(size < capacity);
        new (&storage[size]) T(args ...);
        size += 1;
    }

    template<typename T>
    __host__ __device__
    void RestrainedVector<T>::clear()
    {
        size = 0;
    }

    template<typename T>
    __host__ __device__
    void RestrainedVector<T>::popBack()
    {
        Extra::Utils::Platform::assert(size > 0);
        size -= 1;
    }
};


