#pragma once

#include "BB/AugmentedState.cuh"
#include "DD/MDD.cuh"

template<typename ProblemType,typename StateType>
class OffloadBuffer
{
    // Members
    private:
    unsigned int size;
    Array<StateType> statesBuffer;
    Array<BB::AugmentedState<StateType>> augmentedStates;
    Array<DD::MDD<ProblemType,StateType>> mdds;
    Array<Neighbourhood> neighbourhoods;

    // Functions
    public:
    OffloadBuffer(ProblemType const * problem, unsigned int maxWidth, unsigned int capacity, Memory::MallocType mallocType);
    void clear();
    void enqueue(BB::AugmentedState<StateType> const * augmentedState);
    void generateNeighbourhoods(StateType const * currentSolution, unsigned int eqPercentage, unsigned int neqPercentage, std::mt19937* rng);
    __host__ __device__ void doOffload(unsigned int index);
    DD::MDD<ProblemType,StateType> const * getMDD(unsigned int index) const;
    unsigned int getSize() const;
    bool isEmpty() const;
    bool isFull() const;
};

template<typename ProblemType, typename StateType>
OffloadBuffer<ProblemType,StateType>::OffloadBuffer(ProblemType const * problem, unsigned int maxWidth, unsigned int capacity, Memory::MallocType mallocType) :
    size(0),
    statesBuffer(capacity, mallocType),
    augmentedStates(capacity, mallocType),
    mdds(capacity, mallocType),
    neighbourhoods(capacity, mallocType)
{
    // States
    for (unsigned int stateIdx = 0; stateIdx < statesBuffer.getCapacity(); stateIdx += 1)
    {
        new (statesBuffer[stateIdx]) StateType(problem, Memory::MallocType::Std);
    }

    // Augmented states
    for (unsigned int augmentedStatesIdx = 0; augmentedStatesIdx < augmentedStates.getCapacity(); augmentedStatesIdx += 1)
    {
        augmentedStates[augmentedStatesIdx]->state = statesBuffer[augmentedStatesIdx];
    }

    // MDDs
    for (unsigned int mddIdx = 0; mddIdx < mdds.getCapacity(); mddIdx += 1)
    {
        new (mdds[mddIdx]) DD::MDD<ProblemType,StateType>(problem, maxWidth, mallocType);
    }

    // Neighbourhood
    for (unsigned int neighbourhoodIdx = 0; neighbourhoodIdx < neighbourhoods.getCapacity(); neighbourhoodIdx += 1)
    {
        new (neighbourhoods[neighbourhoodIdx]) Neighbourhood(problem, mallocType);
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void OffloadBuffer<ProblemType, StateType>::doOffload(unsigned int index)
{
    mdds[index]->setTop(augmentedStates[index]->state);
    mdds[index]->buildTopDown(DD::Type::Relaxed, neighbourhoods[index]);
    augmentedStates[index]->lowerbound = mdds[index]->getBottom()->cost;

    mdds[index]->setTop(augmentedStates[index]->state);
    mdds[index]->buildTopDown(DD::Type::Restricted, neighbourhoods[index]);
    augmentedStates[index]->upperbound = mdds[index]->getBottom()->cost;
    augmentedStates[index]->state = mdds[index]->getBottom();
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType,StateType>::clear()
{
    size = 0;
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType,StateType>::enqueue(BB::AugmentedState<StateType> const * augmentedState)
{
    unsigned int const index = size;
    *augmentedStates[index] = *augmentedState;
    size += 1;
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType,StateType>::generateNeighbourhoods(StateType const * currentSolution, unsigned int eqPercentage, unsigned int neqPercentage, std::mt19937* rng)
{
    for (unsigned int index = 0; index < size; index += 1)
    {
        neighbourhoods[index]->generate(&currentSolution->selectedValues, eqPercentage, neqPercentage, rng);
    }
}

template<typename ProblemType, typename StateType>
DD::MDD<ProblemType, StateType> const * OffloadBuffer<ProblemType, StateType>::getMDD(unsigned int index) const
{
    return mdds[index];
}

template<typename ProblemType, typename StateType>
unsigned int OffloadBuffer<ProblemType,StateType>::getSize() const
{
    return size;
}

template<typename ProblemType, typename StateType>
bool OffloadBuffer<ProblemType,StateType>::isEmpty() const
{
    return size == 0;
}

template<typename ProblemType, typename StateType>
bool OffloadBuffer<ProblemType,StateType>::isFull() const
{
    return size == neighbourhoods.getCapacity();
}