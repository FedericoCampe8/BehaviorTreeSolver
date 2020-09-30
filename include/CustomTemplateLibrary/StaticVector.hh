#pragma once

#include <cassert>
#include <algorithm>

#include <CustomTemplateLibrary/Types.hh>

namespace ctl
{

    template<typename T>
    class StaticVector
    {
        public:
            uint const capacity;

        private:
            uint size;
            T * const storage;

        public:
            StaticVector(uint capacity);
            StaticVector(uint capacity, void * const memory);

            T & operator[](uint index) const;
            StaticVector<T> & operator=(StaticVector<T> const & other);
            bool operator==(StaticVector<T> const & other) const;

            uint getSize() const;
            void resize(uint size);

            T & at(uint index) const;
            T & front() const;
            T & back() const;

            T * begin() const;
            T * end() const;

            template<typename ... Args>
            void emplaceBack(Args ... args);
            void popBack();
            void clear();
    };

    template<typename T>
    StaticVector<T>::StaticVector(uint capacity) :
        StaticVector(capacity, malloc(sizeof(T) * capacity))
    {}


    template<typename T>
    StaticVector<T>::StaticVector(uint capacity, void * const memory) :
        capacity(capacity),
        size(0),
        storage(static_cast<T*>(memory))
    {}

    template<typename T>
    T &StaticVector<T>::operator[](uint index) const
    {
        assert(size > 0);
        assert(index < capacity);

        return storage[index];
    }

    template<typename T>
    StaticVector<T> & StaticVector<T>::operator=(StaticVector<T> const & other)
    {
        assert(other.size <= capacity);
        size = other.size;
        std::copy(other.begin(), other.end(), begin());
        return *this;
    }

    template<typename T>
    bool StaticVector<T>::operator==(StaticVector<T> const & other) const
    {
        return size != other.size ? false : std::equal(begin(), end(), other.begin(), other.end());
    }

    template<typename T>
    uint StaticVector<T>::getSize() const
    {
        return size;
    }

    template<typename T>
    void StaticVector<T>::resize(uint size)
    {
        this->size = size;
    }

    template<typename T>
    T & StaticVector<T>::at(uint index) const
    {
        return operator[](index);
    }

    template<typename T>
    T & StaticVector<T>::front() const
    {
        return at(0);
    }

    template<typename T>
    T & StaticVector<T>::back() const
    {
        return at(size - 1);
    }

    template<typename T>
    T * StaticVector<T>::begin() const
    {
        return storage;
    }

    template<typename T>
    T * StaticVector<T>::end() const
    {
        return storage + size;
    }

    template<typename T>
    template<typename ... Args>
    void StaticVector<T>::emplaceBack(Args ... args)
    {
        assert(size < capacity);
        new (&storage[size]) T(args ...);
        size += 1;
    }

    template<typename T>
    void StaticVector<T>::clear()
    {
        size = 0;
    }

    template<typename T>
    void StaticVector<T>::popBack()
    {
        assert(size > 0);
        size -= 1;
    }
};


