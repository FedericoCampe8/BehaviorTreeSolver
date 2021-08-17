#pragma once

#include <DD/MDD.cuh>
#include <LNS/Context.h>
#include <LNS/Neighbourhood.cuh>
#include <LNS/StatesPriorityQueue.cuh>

template<typename ProblemType, typename StateType>
class OffloadBuffer
{
    // Members
    protected:
    Array<StateType> topStates;
    Array<StateType> bottomStates;
    u32 size;
    u32 capacity;
    Array<DD::MDD<ProblemType, StateType>> mdds;
    Array<Neighbourhood> neighbourhoods;

    // Functions
    public:
    OffloadBuffer(ProblemType const * problem, u32 mddsWidth, u32 capacity, float eqProbability, float neqProbability, Memory::MallocType mallocType);
    void initializeOffload(StatesPriorityQueue<StateType>* statesPriorityQueue);
    void initializeOffload(StateType const * statesPriorityQueue);
    void finalizeOffload(StatesPriorityQueue<StateType>* statesPriorityQueue);
    u32 getSize() const;
    StateType const * getBestSolution(LNS::SearchPhase searchPhase) const;
    void printNeighbourhoods() const;
    protected:
    bool isEmpty() const;
    bool isFull() const;
};

template<typename ProblemType, typename StateType>
OffloadBuffer<ProblemType, StateType>::OffloadBuffer(ProblemType const * problem, u32 mddsWidth, u32 capacity, float eqProbability, float neqProbability, Memory::MallocType mallocType) :
    capacity(capacity),
    topStates(capacity, mallocType),
    bottomStates(capacity, mallocType),
    mdds(capacity, mallocType),
    neighbourhoods(capacity, mallocType)
{
    // Top states
    std::byte* storages = StateType::mallocStorages(problem, capacity, mallocType);
    for (u32 stateIdx = 0; stateIdx < capacity; stateIdx += 1)
    {
        new (topStates[stateIdx]) StateType(problem, storages);
        storages = Memory::align(topStates[stateIdx]->endOfStorage(), Memory::DefaultAlignment);
    }

    // Bottom states
    storages = StateType::mallocStorages(problem, capacity, mallocType);
    for (u32 stateIdx = 0; stateIdx < capacity; stateIdx += 1)
    {
        new (bottomStates[stateIdx]) StateType(problem, storages);
        storages = Memory::align(bottomStates[stateIdx]->endOfStorage(), Memory::DefaultAlignment);
    }

    // MDDs
    for (u32 mddIdx = 0; mddIdx < capacity; mddIdx += 1)
    {
        new (mdds[mddIdx]) DD::MDD<ProblemType, StateType>(problem, mddsWidth, mallocType);
    }

    // Neighbourhood
    for (u32 neighbourhoodIdx = 0; neighbourhoodIdx < capacity; neighbourhoodIdx += 1)
    {
        new (neighbourhoods[neighbourhoodIdx]) Neighbourhood(problem, eqProbability, neqProbability, mallocType);
    }
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType, StateType>::initializeOffload(StatesPriorityQueue<StateType>* statesPriorityQueue)
{
    size = 0;
    while (not (statesPriorityQueue->isEmpty() or isFull()))
    {
        *topStates[size] = *statesPriorityQueue->front();
        statesPriorityQueue->popFront();
        size += 1;
    }
}

template<typename ProblemType, typename StateType>
bool OffloadBuffer<ProblemType, StateType>::isFull() const
{
    return size == capacity;
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType, StateType>::initializeOffload(StateType const * state)
{
    if(capacity > 0) //In case of no capacity
    {
        size = 1;
        *topStates[0] = *state;
    }
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType, StateType>::finalizeOffload(StatesPriorityQueue<StateType>* statesPriorityQueue)
{
    for (u32 index = 0; index < size; index += 1)
    {
        Vector<StateType>& cutset = mdds[index]->cutsetStates;
        for (StateType* cutsetState = cutset.begin(); cutsetState != cutset.end(); cutsetState += 1)
        {
            if (not statesPriorityQueue->isFull() and cutsetState->isValid())
            {
                statesPriorityQueue->insert(cutsetState);
            }
        }
    }
}
template<typename ProblemType, typename StateType>
u32 OffloadBuffer<ProblemType, StateType>::getSize() const
{
    return size;
}

template<typename ProblemType, typename StateType>
StateType const * OffloadBuffer<ProblemType, StateType>::getBestSolution(LNS::SearchPhase searchPhase) const
{
    if(not this->isEmpty())
    {
        u32 stateIdxEnd = searchPhase == LNS::SearchPhase::Init ? size : capacity;
        StateType* bestSolution = bottomStates[0];
        for(u32 stateIdx = 1; stateIdx < stateIdxEnd; stateIdx += 1)
        {
            if(bottomStates[stateIdx]->cost < bestSolution->cost)
            {
                bestSolution = bottomStates[stateIdx];
            }
        }
        if(bestSolution->isValid())
        {
            return bestSolution;
        }
        else
        {
            return nullptr;
        }
    }
    else
    {
        return nullptr;
    }
}

template<typename ProblemType, typename StateType>
bool OffloadBuffer<ProblemType, StateType>::isEmpty() const
{
    return size == 0;
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType, StateType>::printNeighbourhoods() const
{
    for(u32 index = 0; index < capacity; index +=1)
    {
        neighbourhoods.at(index)->print(true);
    }
}
