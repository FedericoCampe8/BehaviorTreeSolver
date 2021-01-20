#pragma once
#include "../DP/State.cuh"

namespace BB
{
    template<typename StateType>
    class AugmentedState
    {
        // Members
        public:
        DP::CostType upperbound;
        DP::CostType lowerbound;
        StateType const* state;

        // Functions
        public:
        AugmentedState(StateType const* state);
        AugmentedState(DP::CostType upperbound, DP::CostType lowerbound, StateType const* state);
        __host__ __device__ bool operator<(AugmentedState<StateType>& other) const;
        __host__ __device__ static void swap(AugmentedState<StateType>& as0, AugmentedState<StateType>& as1);
    };
}

template<typename StateType>
BB::AugmentedState<StateType>::AugmentedState(StateType const* state) :
    AugmentedState(DP::MaxCost, 0, state)
{}

template<typename StateType>
BB::AugmentedState<StateType>::AugmentedState(DP::CostType upperbound, DP::CostType lowerbound, StateType const* state) :
    upperbound(upperbound),
    lowerbound(lowerbound),
    state(state)
{}

template<typename StateType>
__host__ __device__
bool BB::AugmentedState<StateType>::operator<(AugmentedState<StateType>& other) const
{
    return upperbound < other.upperbound;
}

template<typename StateType>
__host__ __device__
void BB::AugmentedState<StateType>::swap(AugmentedState<StateType>& as0, AugmentedState<StateType>& as1)
{
    thrust::swap(as0.upperbound, as1.upperbound);
    thrust::swap(as0.lowerbound, as1.lowerbound);
    thrust::swap(as0.state, as1.state);
}