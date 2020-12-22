#pragma once

namespace BB
{
    template<typename StateType>
    class QueuedState
    {
        // Members
        public:
            unsigned int* heapIdx;
            unsigned int lowerbound;
            unsigned int upperbound;
            StateType const * state;

        // Functions
        public:
           __host__ __device__ QueuedState(unsigned int* heapIdx, unsigned int lowerBound, unsigned int upperBound, StateType const * state);
           __host__ __device__ void operator=(QueuedState<StateType> const & other);
    };

    template<typename StateType>
    __host__ __device__
    QueuedState<StateType>::QueuedState(unsigned int* heapIdx, unsigned int lowerBound, unsigned int upperBound, StateType const * state) :
        heapIdx(heapIdx),
        lowerbound(lowerBound),
        upperbound(upperBound),
        state(state)
    {}

    template<typename StateType>
    __host__ __device__
    void QueuedState<StateType>::operator=(QueuedState<StateType> const & other)
    {
        *heapIdx = *other.heapIdx;
        lowerbound = other.lowerbound;
        upperbound = other.upperbound;
        state = other.state;
    }
}


