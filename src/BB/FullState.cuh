#pragma once

#include <cstdint>

namespace BB
{
    template<typename T>
    class FullState
    {
        public:
            bool active;
            uint32_t lowerBound;
            uint32_t upperBound;
            T* state;

        public:
            __host__ __device__ FullState(bool active, T* state);
            __host__ __device__ FullState(bool active, int lowerBound, int upperBound, T* state);
            __host__ __device__ FullState(int lowerBound, int upperBound, T* state);
            __host__ __device__ FullState<T>& operator=(FullState<T> const & other);
            __host__ __device__ void print();

    };


    template<typename T>
    __host__ __device__
    FullState<T>::FullState(bool active,  T* state) :
        active(active),
        lowerBound(0),
        upperBound(UINT32_MAX),
        state(state)
    {}

    template<typename T>
    __host__ __device__
    FullState<T>::FullState(bool active, int lowerBound, int upperBound,  T* state) :
        active(active),
        lowerBound(lowerBound),
        upperBound(upperBound),
        state(state)
    {}

    template<typename T>
    __host__ __device__
    FullState<T>& FullState<T>::operator=(FullState<T> const & other)
    {
        active = other.active;
        lowerBound = other.lowerBound;
        upperBound = other.upperBound;
        *state = *other.state;
        return *this;
    }
    template<typename T>
    __host__ __device__
    void FullState<T>::print()
    {
        printf("Active: %d | LB: %d | UB: %d | ", active ? "T" : "F", lowerBound , upperBound);
        state->print();
    }
}


