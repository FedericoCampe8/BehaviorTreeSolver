#pragma once

#include <cstddef>

#include <Extra/Utils.hh>

namespace Extra::Containers
{
    template<typename T>
    class RestrainedArray
    {
        private:
            bool const owning: 1;
        public:
            uint const size: 31;
        private:
            T * const storage;

        public:
            __host__ __device__ RestrainedArray(uint size);
            __host__ __device__ RestrainedArray(uint size, T * const storage);
            __host__ __device__ RestrainedArray(uint size, std::byte * const storage);
            __host__ __device__ ~RestrainedArray();
            __host__ __device__ T & operator[](uint index) const;
            __host__ __device__ T & at(uint index) const;
            __host__ __device__ T & front() const;
            __host__ __device__ T & back() const;
            __host__ __device__ T * begin() const;
            __host__ __device__ T * end() const;
            __host__ __device__ bool operator==(RestrainedArray<T> const & other) const;
    };

    template<typename T>
    __host__ __device__
    RestrainedArray<T>::RestrainedArray(uint size) :
        owning(true),
        size(size),
        storage(static_cast<T*>(Extra::Utils::Memory::malloc(sizeof(T) * size)))
    {}

    template<typename T>
    __host__ __device__
    RestrainedArray<T>::RestrainedArray(uint size, T * const storage) :
       owning(false),
       size(size),
       storage(storage)
    {}

    template<typename T>
    __host__ __device__
    RestrainedArray<T>::RestrainedArray(uint size, std::byte * const storage) :
        RestrainedArray<T>(size, reinterpret_cast<T*>(storage))
    {}

    template<typename T>
    __host__ __device__ RestrainedArray<T>::~RestrainedArray()
    {
        if(owning)
        {
            Extra::Utils::Memory::free(storage);
        }
    }

    template<typename T>
    __host__ __device__
    T & RestrainedArray<T>::operator[](uint index) const
    {
        Extra::Utils::Platform::assert(index < size);
        return storage[index];
    }

    template<typename T>
    __host__ __device__
    T & RestrainedArray<T>::at(uint index) const
    {
        return operator[](index);
    }

    template<typename T>
    __host__ __device__
    T & RestrainedArray<T>::front() const
    {
        return at(0);
    }

    template<typename T>
    __host__ __device__
    T & RestrainedArray<T>::back() const
    {
        return at(size - 1);
    }

    template<typename T>
    __host__ __device__
    T * RestrainedArray<T>::begin() const
    {
        return storage;
    }

    template<typename T>
    __host__ __device__
    T * RestrainedArray<T>::end() const
    {
        return storage + size;
    }

    template<typename T>
    __host__ __device__
    bool RestrainedArray<T>::operator==(RestrainedArray<T> const & other) const
    {
        return size == other.size and Extra::Algorithms::equal(begin(), end(), other.begin());

    }
};


