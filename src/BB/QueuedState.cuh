#pragma once

namespace BB
{
    template<typename StateType>
    class QueuedState
    {
        // Members
        public:
            unsigned int * heapIdx;
            unsigned int lowerbound;
            unsigned int upperbound;
            StateType const * state;

        // Functions
        public:
           __host__ __device__ QueuedState(unsigned int* heapIdx, unsigned int lowerBound, unsigned int upperBound, StateType const * state);
           __host__ __device__ static void swap(QueuedState<StateType>* queuedState0,  QueuedState<StateType>* queuedState1);
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
    void QueuedState<StateType>::swap(QueuedState<StateType>* queuedState0, QueuedState<StateType>* queuedState1)
    {
        unsigned int heapIdx0 = *queuedState0->heapIdx;
        *queuedState0->heapIdx = *queuedState1->heapIdx;
        *queuedState1->heapIdx = heapIdx0;

        thrust::swap(queuedState0->heapIdx, queuedState1->heapIdx);
        thrust::swap(queuedState0->lowerbound, queuedState1->lowerbound);
        thrust::swap(queuedState0->upperbound, queuedState1->upperbound);
        thrust::swap(queuedState0->state, queuedState1->state);
    }
}


