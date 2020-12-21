#pragma once

namespace BB
{
    template<typename StateType>
    class QueuedState
    {
        public:
            // Queue information
            unsigned int* heapIdx;

            // Search information
            unsigned int lowerbound;
            unsigned int upperbound;

            // State
            StateType& state;

        public:
           QueuedState(unsigned int* heapIdx, unsigned int lowerBound, unsigned int upperBound, StateType& state);
           QueuedState<StateType>& operator=(QueuedState<StateType> const & other);
    };

    template<typename StateType>
    QueuedState<StateType>::QueuedState(unsigned int* heapIdx, unsigned int lowerBound, unsigned int upperBound, StateType& state) :
        heapIdx(heapIdx),
        lowerbound(lowerBound),
        upperbound(upperBound),
        state(state)
    {}

    template<typename StateType>
    QueuedState<StateType>& QueuedState<StateType>::operator=(QueuedState<StateType> const & other)
    {
        *heapIdx = *other.heapIdx;
        lowerbound = other.lowerbound;
        upperbound = other.upperbound;
        state = other.state;
        return *this;
    }
}


