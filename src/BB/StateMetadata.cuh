#pragma once

#include <cstdint>
#include <thrust/swap.h>

namespace BB
{
    template<typename StateType>
    class StateMetadata
    {
        // Members
        public:
            uint32_t lowerbound;
            uint32_t upperbound;
            StateType const * state;

        // Functions
        public:
            StateMetadata(unsigned int lowerbound, unsigned int upperbound, StateType const * state);
            StateMetadata(StateMetadata const * other);
            __host__ __device__ static void swap(StateMetadata* stateMetadata0, StateMetadata* stateMetadata1);
    };

    template<typename StateType>
    StateMetadata<StateType>::StateMetadata(unsigned int lowerbound, unsigned int upperbound, const StateType* state) :
        lowerbound(lowerbound),
        upperbound(upperbound),
        state(state)
    {}

    template<typename StateType>
    StateMetadata<StateType>::StateMetadata(StateMetadata const * other) :
        lowerbound(other->lowerbound),
        upperbound(other->upperbound),
        state(other->state)
    {}

    template<typename StateType>
    void StateMetadata<StateType>::swap(StateMetadata* stateMetadata0, StateMetadata* stateMetadata1)
    {
        thrust::swap(stateMetadata0->lowerbound, stateMetadata1->lowerbound);
        thrust::swap(stateMetadata0->upperbound, stateMetadata1->upperbound);
        thrust::swap(stateMetadata0->state, stateMetadata1->state);
    }
}