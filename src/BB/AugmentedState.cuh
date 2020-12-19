#pragma once

#include <cstdint>

namespace BB
{
    template<typename T>
    class AugmentedState
    {
        public:
            // Queue information
            unsigned int* heapIdx;

            // Search information
            unsigned int lowerBound;
            unsigned int upperBound;
            unsigned int cost;
            unsigned int selectedValuesCount;

            // State
            T* state;

        public:
            __host__ __device__ AugmentedState(unsigned int* heapIdx, unsigned int lowerBound, unsigned int upperBound, T* state);
            __host__ __device__ AugmentedState(AugmentedState<T> const & other);
            __host__ __device__ AugmentedState<T>& operator=(AugmentedState<T> const & other);
    };

    template<typename T>
    __host__ __device__
    AugmentedState<T>::AugmentedState(unsigned int* heapIdx, unsigned int lowerBound, unsigned int upperBound, T* state) :
        heapIdx(heapIdx),
        lowerBound(lowerBound),
        upperBound(upperBound),
        cost(state->cost),
        selectedValuesCount(state->selectedValues().getSize()),
        state(state)
    {}

    template<typename T>
    __host__ __device__
    AugmentedState<T>::AugmentedState(AugmentedState<T> const & other) :
        lowerBound(other.lowerBound),
        upperBound(other.upperBound),
        cost(other.cost),
        selectedValuesCount(other.selectedValuesCount),
        state(other.state)
    {
        *heapIdx = *other.heapIdx;
    }

    template<typename T>
    __host__ __device__
    AugmentedState<T>& AugmentedState<T>::operator=(AugmentedState<T> const & other)
    {
        lowerBound = other.lowerBound;
        upperBound = other.upperBound;
        cost = other.cost;
        selectedValuesCount = other.selectedValuesCount;
        *heapIdx = *other.heapIdx;
        state = other.state;
        return *this;
    }
}


