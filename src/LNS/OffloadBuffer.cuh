#pragma once

#include <DD/MDD.cuh>
#include <LNS/Context.h>
#include <LNS/Neighbourhood.cuh>
#include <LNS/StatesPriorityQueue.cuh>
#include <LNS/SyncState.cuh>

template<typename ProblemType, typename StateType>
class OffloadBuffer
{
    // Members
    private:
    u32 size;
    u32 capacity;
    Array<StateType> topStates;
    Array<StateType> bottomStates;
    Array<DD::MDD<ProblemType, StateType>> mdds;
    Array<Neighbourhood> neighbourhoods;

    // Functions
    public:
    OffloadBuffer(ProblemType const * problem, u32 mddsWidth, u32 capacity, float eqProbability, float neqProbability, Memory::MallocType mallocType);
    void initializeOffload(StatesPriorityQueue<StateType>* statesPriorityQueue);
    void initializeOffload(StateType const * statesPriorityQueue);
    __host__ __device__ void doOffload(LNS::SearchPhase searchPhase, u32 index);
    void finalizeOffload(StatesPriorityQueue<StateType>* statesPriorityQueue);
    void getBestSolution(LNS::SearchPhase searchPhase, SyncState<ProblemType, StateType> * solution);
    void generateNeighborhood(Array<std::mt19937>* rngs, Vector<OP::ValueType> * values, u32 index);
    __device__ void generateNeighborhood(Array<curandState>* rngs, Vector<OP::ValueType> * values, u32 index);
    void printNeighborhoods() const;
    bool isEmpty() const;
    u32 getSize() const;
    u32 getCapacity() const;
    private:
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
    statesPriorityQueue->mutex.lock();

    size = 0;
    while (not (statesPriorityQueue->isEmpty() or isFull()))
    {
        *topStates[size] = *statesPriorityQueue->front();
        statesPriorityQueue->popFront();
        size += 1;
    }

    statesPriorityQueue->mutex.unlock();
}

template<typename ProblemType, typename StateType>
bool OffloadBuffer<ProblemType, StateType>::isFull() const
{
    return size == capacity;
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType, StateType>::initializeOffload(StateType const * state)
{
    size = 1;
    *topStates[0] = *state;
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType, StateType>::finalizeOffload(StatesPriorityQueue<StateType>* statesPriorityQueue)
{
    statesPriorityQueue->mutex.lock();

    for (u32 index = 0; index < size; index += 1)
    {
        if(topStates[index]->selectedValues.getSize() < topStates[index]->selectedValues.getCapacity() - 1)
        {
            Vector<StateType> const * cutset = &mdds[index]->cutsetStates;
            for (StateType* cutsetState = cutset->begin(); cutsetState != cutset->end(); cutsetState += 1)
            {
                if (cutsetState->isValid() and (not statesPriorityQueue->isFull()))
                {
                    statesPriorityQueue->insert(cutsetState);
                }
            }
        }
    }

    statesPriorityQueue->mutex.unlock();
}
template<typename ProblemType, typename StateType>
u32 OffloadBuffer<ProblemType, StateType>::getSize() const
{
    return size;
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType, StateType>::getBestSolution(LNS::SearchPhase searchPhase, SyncState<ProblemType, StateType> * solution)
{
    solution->mutex.lock();
    solution->state.invalidate();
    u32 stateIdxEnd = searchPhase == LNS::SearchPhase::Init ? size : capacity;
    for(u32 stateIdx = 0; stateIdx < stateIdxEnd; stateIdx += 1)
    {
        if(bottomStates[stateIdx]->cost < solution->state.cost)
        {
            solution->state = *bottomStates[stateIdx];
        }
    }
    solution->mutex.unlock();
}

template<typename ProblemType, typename StateType>
bool OffloadBuffer<ProblemType, StateType>::isEmpty() const
{
    return size == 0;
}

template<typename ProblemType, typename StateType>
u32 OffloadBuffer<ProblemType, StateType>::getCapacity() const
{
    return capacity;
}

template<typename ProblemType, typename StateType>
__host__ __device__
void OffloadBuffer<ProblemType, StateType>::doOffload(LNS::SearchPhase searchPhase, u32 index)
{
    if(searchPhase == LNS::SearchPhase::Init)
    {
        mdds[index]->buildTopDown(neighbourhoods[index], topStates[index], bottomStates[index], false);
    }
    else
    {
        mdds[index]->buildTopDown(neighbourhoods[index], topStates[0], bottomStates[index], true);
    }
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType, StateType>::generateNeighborhood(Array<std::mt19937>* rngs, Vector<OP::ValueType> * values, u32 index)
{
    neighbourhoods[index]->generate(rngs->at(index), values);
}

template<typename ProblemType, typename StateType>
__device__
void OffloadBuffer<ProblemType, StateType>::generateNeighborhood(Array<curandState>* rngs, Vector<OP::ValueType>* values, u32 index)
{
    neighbourhoods[index]->generate(rngs->at(index), values);
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType, StateType>::printNeighborhoods() const
{
    for(Neighbourhood* neighbourhood = neighbourhoods.begin(); neighbourhood != neighbourhoods.end(); neighbourhood +=1)
    {
        neighbourhood->print(true);
    }
}