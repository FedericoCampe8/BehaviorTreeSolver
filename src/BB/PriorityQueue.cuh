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
        MaxHeap<BB::AugmentedState<StateType>> maxHeap;

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
    maxHeap(capacity, Memory::MallocType::Std)
{
    // States
    std::byte* storages = StateType::mallocStorages(problem, capacity, Memory::MallocType::Std);
    for (u32 stateIdx = 0; stateIdx < statesBuffer.getCapacity(); stateIdx += 1)
    {
        new (statesBuffer[stateIdx]) StateType(problem, storages);
        storages = Memory::align(statesBuffer[stateIdx]->endOfStorage(), Memory::DefaultAlignment);
    }
}

template<typename StateType>
BB::AugmentedState<StateType> const * BB::PriorityQueue<StateType>::front() const
{
    return maxHeap.front();
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
    maxHeap.insert(&bufferedAugmentedState);
}

template<typename StateType>
bool BB::PriorityQueue<StateType>::isEmpty() const
{
    return maxHeap.isEmpty();
}

template<typename StateType>
bool BB::PriorityQueue<StateType>::isFull() const
{
    return maxHeap.isFull();
}

template<typename StateType>
void BB::PriorityQueue<StateType>::popFront()
{
    AugmentedState<StateType> const * const front = maxHeap.front();
    statesBuffer.erase(front->state);
    maxHeap.erase(front);
}