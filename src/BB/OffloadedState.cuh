#pragma once

#include <Containers/Vector.cuh>

#include "StateMetadata.cuh"

namespace BB
{
    template<typename StateType>
    class OffloadedState : public StateMetadata<StateType>
    {
        public:
            LightVector<StateType> cutset;
            StateType* const upperboundState;

        public:
            OffloadedState(StateType const * state, unsigned int cutsetMaxSize, StateType* cutsetBuffer, StateType* upperboundState);
    };

    template<typename StateType>
    OffloadedState<StateType>::OffloadedState(StateType const * state, unsigned int cutsetMaxSize, StateType* cutsetBuffer, StateType* upperboundState) :
        StateMetadata<StateType>(0,0,state),
        cutset(cutsetMaxSize, cutsetBuffer),
        upperboundState(upperboundState)
    {}
}


