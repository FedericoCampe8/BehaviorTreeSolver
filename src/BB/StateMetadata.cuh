#pragma once

#include <thrust/swap.h>

namespace BB
{
    template<typename StateType>
    class StateMetadata
    {
        // Members
        public:
            DP::CostType lowerbound;
            DP::CostType upperbound;
            StateType const * state;

        // Functions
        public:
            StateMetadata(DP::CostType lowerbound, DP::CostType upperbound, StateType const * state);
            __host__ __device__ void operator=(StateMetadata const & other);
            __host__ __device__ static void swap(StateMetadata& sm0, StateMetadata& sm1);
    };

    template<typename StateType>
    StateMetadata<StateType>::StateMetadata(DP::CostType lowerbound, DP::CostType upperbound, const StateType* state) :
        lowerbound(lowerbound),
        upperbound(upperbound),
        state(state)
    {}

    template<typename StateType>
    __host__ __device__
    void StateMetadata<StateType>::operator=(StateMetadata const & other)
    {
        lowerbound = other.lowerbound,
        upperbound = other.upperbound,
        state = other.state;
    }

    template<typename StateType>
    __host__ __device__
    void StateMetadata<StateType>::swap(StateMetadata& sm0, StateMetadata& sm1)
    {
        thrust::swap(sm0.lowerbound, sm1.lowerbound);
        thrust::swap(sm0.upperbound, sm1.upperbound);
        thrust::swap(sm0.state, sm1.state);
    }
}