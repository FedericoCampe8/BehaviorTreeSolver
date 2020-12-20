#pragma once

#include <Containers/Vector.cuh>

namespace BB
{
    template<typename T>
    class OffloadedState
    {
        public:
            // Search information
            unsigned int lowerbound;
            unsigned int upperbound;
            Vector<T> cutset;
            T& upperboundState;

            // State
            T const & state;

        public:
            OffloadedState(unsigned int cutsetMaxSize, T* cutsetBuffer, T& upperbound, T const & state);

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


