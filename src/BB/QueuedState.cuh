#pragma once

namespace BB
{
    template<typename T>
    class QueuedState
    {
        public:
            // Queue information
            unsigned int* heapIdx;

            // Search information
            unsigned int lowerbound;
            unsigned int upperbound;

            // State
            T& state;

        public:
           QueuedState(unsigned int* heapIdx, unsigned int lowerBound, unsigned int upperBound, T const & state);
           QueuedState<T>& operator=(QueuedState<T> const & other);
    };

    template<typename T>
    QueuedState<T>::QueuedState(unsigned int* heapIdx, unsigned int lowerBound, unsigned int upperBound, T const & state) :
        heapIdx(heapIdx),
        lowerbound(lowerBound),
        upperbound(upperBound),
        cost(state.cost),
        selectedValuesCount(state.selectedValues().getSize()),
        state(state)
    {}

    template<typename T>
    QueuedState<T>& QueuedState<T>::operator=(QueuedState<T> const & other)
    {
        *heapIdx = *other.heapIdx;
        lowerbound = other.lowerbound;
        upperbound = other.upperbound;
        cost = other.cost;
        selectedValuesCount = other.selectedValuesCount;
        state = other.state;
        return *this;
    }
}


