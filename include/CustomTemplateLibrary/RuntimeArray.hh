#pragma once

#include <cassert>
#include <algorithm>

#include <CustomTemplateLibrary/Types.hh>

namespace ctl
{

    template<typename T>
    class RuntimeArray
    {
        public:
            uint const size;

        private:
            T * const storage;

        public:
            RuntimeArray(uint size);
            RuntimeArray(uint size, void * const storage);


            T & operator[](uint index) const;
            T & at(uint index) const;
            T & front() const;
            T & back() const;

            T * begin() const;
            T * end() const;

            bool operator==(RuntimeArray<T> const & other) const;
    };


    template<typename T>
    RuntimeArray<T>::RuntimeArray(uint size) :
        RuntimeArray(size, malloc(sizeof(T) * size))
    {}

    template<typename T>
    RuntimeArray<T>::RuntimeArray(uint size, void * const storage) :
       size(size),
       storage(static_cast<T*>(storage))
    {}

    template<typename T>
    T & RuntimeArray<T>::operator[](uint index) const
    {
        assert(index < size);
        return storage[index];
    }

    template<typename T>
    T & RuntimeArray<T>::at(uint index) const
    {
        return operator[](index);
    }

    template<typename T>
    T & RuntimeArray<T>::front() const
    {
        return at(0);
    }

    template<typename T>
    T & RuntimeArray<T>::back() const
    {
        return at(size - 1);
    }

    template<typename T>
    T * RuntimeArray<T>::begin() const
    {
        return storage;
    }

    template<typename T>
    T * RuntimeArray<T>::end() const
    {
        return storage + size;
    }

    template<typename T>
    bool RuntimeArray<T>::operator==(RuntimeArray<T> const & other) const
    {
        return size == other.size ? std::equal(begin(), end(), other.begin(), other.end()) : false;
    }
};


