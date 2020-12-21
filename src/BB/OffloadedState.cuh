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
            Vector<StateType> cutset;
            StateType& upperboundState;

            // State
            StateType const & state;

        public:
            OffloadedState(unsigned int cutsetMaxSize, StateType* cutsetBuffer, StateType& upperbound, StateType const & state);

    };

    template<typename T>
    OffloadedState<T>::OffloadedState(unsigned int cutsetMaxSize, T* cutsetBuffer, T& upperboundState, T const & state) :
        lowerbound(0u),
        upperbound(0u),
        cutset(cutsetBuffer, cutsetBuffer + cutsetMaxSize),
        upperboundState(upperboundState),
        state(state)
    {}
}


