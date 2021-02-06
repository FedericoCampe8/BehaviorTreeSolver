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
        StateType const * state;

        // Functions
        public:
        AugmentedState(StateType const * state);
        AugmentedState(DP::CostType upperbound, DP::CostType lowerbound, StateType const * state);
        __host__ __device__ bool operator<(AugmentedState<StateType>& other) const;
        __host__ __device__ void print(bool endLine = true) const;
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
bool BB::AugmentedState<StateType>::operator<(AugmentedState<StateType>& other) const
{
    /*
    u32 const l0 = state->selectedValues.getSize();
    u32 const l1 = other.state->selectedValues.getSize();
    if(l0 < l1)
    {
        return true;
    }
    else if (l0 == l1)
    {
        return lowerbound < other.lowerbound;
    }
    else
    {
        return false;
    }
     */
        return state->cost < other.state->cost;
}

template<typename StateType>
__host__ __device__
void BB::AugmentedState<StateType>::print(bool endLine) const
{
    printf("Lowerbound: %u | Cost: %u | Upperbound: %u", lowerbound, state->cost, upperbound);
    printf(endLine ? "\n" : "");
}

template<typename StateType>
__host__ __device__
void BB::AugmentedState<StateType>::swap(AugmentedState<StateType>& as0, AugmentedState<StateType>& as1)
{
    thrust::swap(as0.upperbound, as1.upperbound);
    thrust::swap(as0.lowerbound, as1.lowerbound);
    thrust::swap(as0.state, as1.state);
}