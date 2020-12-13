#pragma once

#include <cstdint>

namespace BB
{
    template<typename T>
    class AugmentedState
    {
        public:
            unsigned int lowerBound;
            unsigned int upperBound;
            T* const state;

        public:
            __host__ __device__ AugmentedState(T* state);
            __host__ __device__ AugmentedState<T>& operator=(AugmentedState<T> const & other);

    };

    template<typename T>
    __host__ __device__
    AugmentedState<T>::AugmentedState(T* state) :
        lowerBound(INT32_MIN),
        upperBound(INT32_MAX),
        state(state)
    {}

    template<typename T>
    __host__ __device__
    AugmentedState<T>& AugmentedState<T>::operator=(AugmentedState<T> const & other)
    {
        lowerBound = other.lowerBound;
        upperBound = other.upperBound;
        *state = *other.state;
        return *this;
    }
}


