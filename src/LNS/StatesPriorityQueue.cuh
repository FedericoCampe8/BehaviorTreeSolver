#pragma once

#include "../../include/Containers/Buffer.cuh"
#include "../../include/Containers/MinHeap.cuh"


template<typename StateType>
class StatesPriorityQueue
{
    // Members
    private:
    Buffer<StateType> statesBuffer;
    MinHeap<StateType*> minHeap;

    // Functions
    public:
    template<typename ProblemType>
    StatesPriorityQueue(ProblemType const * problem, u32 capacity);
    bool isEmpty() const;
    StateType const * front() const;
    void popFront();
    void insert(StateType const * state);
    bool isFull() const;
    private:
    u32 getSize() const;


};

template<typename StateType>
template<typename ProblemType>
StatesPriorityQueue<StateType>::StatesPriorityQueue(ProblemType const * problem, u32 capacity) :
    statesBuffer(capacity, Memory::MallocType::Std),
    minHeap(capacity, Memory::MallocType::Std)
{
    // States
    std::byte* storage = StateType::mallocStorages(problem, capacity, Memory::MallocType::Std);
    for (u32 stateIdx = 0; stateIdx < statesBuffer.getCapacity(); stateIdx += 1)
    {
        new (statesBuffer[stateIdx]) StateType(problem, storage);
        storage = Memory::align(statesBuffer[stateIdx]->endOfStorage(), Memory::DefaultAlignment);
    }
}

template<typename StateType>
StateType const * StatesPriorityQueue<StateType>::front() const
{
    return *minHeap.front();
}

template<typename StateType>
u32 StatesPriorityQueue<StateType>::getSize() const
{
    return statesBuffer.getSize();
}

template<typename StateType>
void StatesPriorityQueue<StateType>::insert(StateType const * state)
{
    StateType *  bufferedState = statesBuffer.insert(state);
    minHeap.insert(&bufferedState);
}

template<typename StateType>
bool StatesPriorityQueue<StateType>::isEmpty() const
{
    return minHeap.isEmpty();
}

template<typename StateType>
bool StatesPriorityQueue<StateType>::isFull() const
{
    return minHeap.isFull();
}

template<typename StateType>
void StatesPriorityQueue<StateType>::popFront()
{
    StateType** const front = minHeap.front();
    statesBuffer.erase(*front);
    minHeap.popFront();
}

