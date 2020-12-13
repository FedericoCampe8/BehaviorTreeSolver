#pragma once

#include <cstdint>

namespace BB
{
    template<typename T>
    class AugmentedState
    {
        public:
            int lowerBound;
            int upperBound;
            T* const state;

        public:
            __host__ __device__ AugmentedState(T* state);
            __host__ __device__ AugmentedState(int lowerBound, int upperBopund, T* state);
            __host__ __device__ AugmentedState<T>& operator=(AugmentedState<T> const & other);

    };

    template<typename T>
    __host__ __device__
    AugmentedState<T>::AugmentedState(T* state) :
        lowerBound(INT_MIN),
        upperBound(INT_MAX),
        state(state)
    {}

    template<typename T>
    __host__ __device__
    AugmentedState<T>::AugmentedState(int lowerBound, int upperBound, T* state) :
        lowerBound(lowerBound),
        upperBound(upperBound),
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


