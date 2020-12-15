#pragma once

#include <cstdint>

namespace BB
{
    template<typename T>
    class AugmentedState
    {
        public:
            T::CostType lowerBound;
            T::CostType upperBound;
            T* state;

        public:
            __host__ __device__ AugmentedState(AugmentedState<T>&& other);
            __host__ __device__ AugmentedState(T::CostType lowerBound, T::CostType upperBound, T* state);
            __host__ __device__ AugmentedState<T>& operator=(AugmentedState<T> const & other);
            __host__ __device__ AugmentedState<T>& operator=(AugmentedState<T>&& other);

    };

    template<typename T>
    __host__ __device__
    AugmentedState<T>::AugmentedState(AugmentedState<T>&& other) :
        lowerBound(other.lowerBound),
        upperBound(other.upperBound),
        state(other.state)
    {}

    template<typename T>
    __host__ __device__
    AugmentedState<T>::AugmentedState(T::CostType lowerBound, T::CostType upperBound, T* state) :
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

    template<typename T>
    __host__ __device__
    AugmentedState<T>& AugmentedState<T>::operator=(AugmentedState<T>&& other)
    {
        lowerBound = other.lowerBound;
        upperBound = other.upperBound;
        state = other.state;
        return *this;
    }
}


