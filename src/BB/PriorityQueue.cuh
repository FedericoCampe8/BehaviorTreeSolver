#pragma once

#include <Containers/Buffer.cuh>
#include <Containers/MaxHeap.cuh>
#include "AugmentedState.cuh"

namespace BB
{
    template<typename StateType>
    class PriorityQueue
    {
        // Members
        private:
        Buffer<StateType> statesBuffer;
        MaxHeap<BB::AugmentedState<StateType>> heap;

        // Functions
        public:
        template<typename ProblemType>
        PriorityQueue(ProblemType const * problem, unsigned int capacity);
        AugmentedState<StateType> const * front() const;
        unsigned int getSize() const;
        void insert(AugmentedState<StateType> const * augmentedState);
        bool isEmpty() const;
        bool isFull() const;
        void popFront();
    };
}

template<typename StateType>
template<typename ProblemType>
BB::PriorityQueue<StateType>::PriorityQueue(ProblemType const * problem, unsigned int capacity) :
    statesBuffer(capacity, Memory::MallocType::Std),
    heap(capacity, Memory::MallocType::Std)
{
    // States
    for (unsigned int stateIdx = 0; stateIdx < statesBuffer.getCapacity(); stateIdx += 1)
    {
        new (statesBuffer[stateIdx]) StateType(problem, Memory::MallocType::Std);
    }
}

template<typename StateType>
BB::AugmentedState<StateType> const * BB::PriorityQueue<StateType>::front() const
{
    return heap.front();
}

template<typename StateType>
unsigned int BB::PriorityQueue<StateType>::getSize() const
{
    return statesBuffer.getSize();
}

template<typename StateType>
void BB::PriorityQueue<StateType>::insert(AugmentedState<StateType> const * augmentedState)
{
    StateType const * const bufferedState = statesBuffer.insert(augmentedState->state);
    AugmentedState<StateType> const bufferedAugmentedState(augmentedState->upperbound, augmentedState->lowerbound, bufferedState);
    heap.pushBack(&bufferedAugmentedState);
}

template<typename StateType>
bool BB::PriorityQueue<StateType>::isEmpty() const
{
    return heap.isEmpty();
}

template<typename StateType>
bool BB::PriorityQueue<StateType>::isFull() const
{
    return heap.isFull();
}

template<typename StateType>
void BB::PriorityQueue<StateType>::popFront()
{
    AugmentedState<StateType> const* const front = heap.front();
    statesBuffer.erase(front->state);
    heap.erase(front);
}