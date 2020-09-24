#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>

#include <CustomTemplateLibrary/Types.hh>

namespace ctl
{

    template<typename T>
    class Vector
    {
        public:
            uint const maxSize;

        private:
            uint currentSize;
            T data[];

        public:
            static size_t getMemSize(uint maxSize);

            Vector(uint maxSize);
            Vector(Vector<T> const * other);

            size_t getMemSize() const;

            uint const size() const;

            T const * begin() const;
            T* begin();
            T const * end() const;
            T* end();

            T const & at(uint index) const;
            T& at(uint index);
            T const & back() const;
            T& back();

            template<typename ... Args>
            void emplaceBack(Args ... args);
            void pushBack(T &element);
            void popBack();

    };

    template<typename T>
    size_t Vector<T>::getMemSize(uint maxSize)
    {
        return sizeof(Vector<T>) + sizeof(T) * maxSize;
    }

    template<typename T>
    Vector<T>::Vector(uint maxSize) :
        maxSize(maxSize),
        currentSize(0)
    {
    }

    template<typename T>
    Vector<T>::Vector(Vector<T> const * other) :
        maxSize(other->maxSize),
        currentSize(other->currentSize)
    {
        memcpy(&data, &other->data, sizeof(T) * currentSize);
    }

    template<typename T>
    size_t Vector<T>::getMemSize() const
    {
        return sizeof(Vector<T>) + sizeof(T) * maxSize;
    }

    template<typename T>
    uint const Vector<T>::size() const
    {
        return currentSize;
    }

    template<typename T>
    T* Vector<T>::begin()
    {
        return const_cast<T*>(const_cast<const Vector<T>*>(this)->begin());
    }

    template<typename T>
    T const * Vector<T>::begin() const
    {
        assert(currentSize > 0);
        return data;
    }

    template<typename T>
    T* Vector<T>::end()
    {
        return const_cast<T*>(const_cast<const Vector<T>*>(this)->end());
    }

    template<typename T>
    T const * Vector<T>::end() const
    {
        assert(currentSize > 0);
        return &data[currentSize + 1];
    }

    template<typename T>
    T& Vector<T>::at(uint index)
    {
        return const_cast<T&>(const_cast<const Vector<T>*>(this)->at(index));
    }

    template<typename T>
    T const & Vector<T>::at(uint index) const
    {
        assert(currentSize > 0);
        assert(index < maxSize);

        return data[index];
    }

    template<typename T>
    T& Vector<T>::back()
    {
        return const_cast<T&>(const_cast<const Vector<T>*>(this)->back());
    }

    template<typename T>
    T const & Vector<T>::back() const
    {
        assert(currentSize > 0);
        return data[currentSize - 1];
    }

    template<typename T>
    template<typename ... Args>
    void Vector<T>::emplaceBack(Args ... args)
    {
        assert(currentSize < maxSize);
        new (&data[currentSize]) T(args ...);
        currentSize += 1;
    }

    template<typename T>
    void Vector<T>::pushBack(T& element)
    {
        assert(currentSize < maxSize);
        data[currentSize] = element;
        currentSize += 1;
    }

    template<typename T>
    void Vector<T>::popBack()
    {
        assert(currentSize > 0);
        currentSize -= 1;
    }
};


