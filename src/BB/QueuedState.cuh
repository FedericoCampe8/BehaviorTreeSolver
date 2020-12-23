#pragma once

#include "StateMetadata.cuh"

namespace BB
{
    template<typename StateType>
    class QueuedState : public StateMetadata<StateType>
    {
        // Members
        public:
            QueuedState** toThis;

        // Functions
        public:
           QueuedState(StateMetadata<StateType> const * stateMetadata, QueuedState** toThis);
           __host__ __device__ static void swap(QueuedState<StateType>* queuedState0,  QueuedState<StateType>* queuedState1);
    };

    template<typename StateType>
    QueuedState<StateType>::QueuedState(StateMetadata<StateType> const * stateMetadata, QueuedState** toThis) :
        StateMetadata<StateType>(stateMetadata),
        toThis(toThis)
    {}

    template<typename StateType>
    void QueuedState<StateType>::swap(QueuedState<StateType>* queuedState0, QueuedState<StateType>* queuedState1)
    {
        StateMetadata<StateType>::swap(queuedState0, queuedState1);

        thrust::swap(queuedState0->toThis, queuedState1->toThis);
        *queuedState0->toThis = queuedState0;
        *queuedState1->toThis = queuedState1;
    }
}


