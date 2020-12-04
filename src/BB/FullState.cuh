#pragma once

#include <cstdint>

namespace BB
{
    template<typename T>
    class FullState
    {
        public:
            int32_t lowerBound;
            int32_t upperBound;
            T* state;

        public:
            FullState(int lowerBound, int upperBound, T* state);
            FullState<T>& operator=(FullState<T> const & other);

    };

    template<typename T>
    FullState<T>::FullState(int lowerBound, int upperBound,  T* state) :
        lowerBound(lowerBound),
        upperBound(upperBound),
        state(state)
    {}

    template<typename T>
    FullState<T>& FullState<T>::operator=(FullState<T> const & other)
    {
        lowerBound = other.lowerBound;
        upperBound = other.upperBound;
        *state = *other.state;
        return *this;
    }
}


