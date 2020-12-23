#pragma once

#include <Containers/Vector.cuh>

namespace BB
{
    template<typename StateType>
    class OffloadedState
    {
        public:
            // Search information
            unsigned int lowerbound;
            unsigned int upperbound;
            LightVector<StateType> cutset;
            StateType* const upperboundState;

            // State
            StateType const * const state;

        public:
            OffloadedState(unsigned int cutsetMaxSize, StateType* cutsetBuffer, StateType* upperbound, StateType const * state);
    };

    template<typename StateType>
    OffloadedState<StateType>::OffloadedState(unsigned int cutsetMaxSize, StateType* cutsetBuffer, StateType* upperboundState, StateType const * state) :
        lowerbound(0u),
        upperbound(0u),
        cutset(cutsetMaxSize, cutsetBuffer),
        upperboundState(upperboundState),
        state(state)
    {}
}


