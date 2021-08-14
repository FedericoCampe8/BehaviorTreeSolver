#pragma once

#include "BB/AugmentedState.cuh"
#include "DD/MDD.cuh"

template<typename ProblemType, typename StateType>
class OffloadBuffer
{
    // Members
    private:
    unsigned int size;
    Array<StateType> statesBuffer;
    Array<DD::MDD<ProblemType, StateType>> mdds;
    Array<Neighbourhood> neighbourhoods;

    // Functions
    public:
    OffloadBuffer(ProblemType const * problem, unsigned int width, unsigned int capacity, Memory::MallocType mallocType);
    __host__ __device__ void doOffload(unsigned int index);
    void clear();
    void enqueue(StateType const * state);
    void generateNeighbourhoods(StateType const* currentSolution, unsigned int eqPercentage, unsigned int neqPercentage, Array<std::mt19937>* rngs, Vector<std::thread>* cpuThreads);
    BB::AugmentedState<StateType> const* getAugmentedState(unsigned int index) const;
    DD::MDD<ProblemType, StateType> const* getMDD(unsigned int index) const;
    unsigned int getSize() const;
    bool isEmpty() const;
    bool isFull() const;
    protected:
    void generateNeighbourhoodsLoop(StateType const* currentSolution, unsigned int eqPercentage, unsigned int neqPercentage, Array<std::mt19937>* rngs, unsigned int begin, unsigned int step, unsigned int end);
    __host__ __device__ inline void barrier() const;
    __host__ __device__ void setTop(unsigned int index);

};

template<typename ProblemType, typename StateType>
OffloadBuffer<ProblemType, StateType>::OffloadBuffer(ProblemType const * problem, unsigned int width, unsigned int capacity, Memory::MallocType mallocType) :
    size(0),
    augmentedStates(capacity, mallocType),
    statesBuffer(capacity, mallocType),
    mdds(capacity, mallocType),
    neighbourhoods(capacity, mallocType)
{
    // States
    std::byte* storages = StateType::mallocStorages(problem, capacity, mallocType);
    for (u32 stateIdx = 0; stateIdx < statesBuffer.getCapacity(); stateIdx += 1)
    {
        new (statesBuffer[stateIdx]) StateType(problem, storages);
        storages = Memory::align(statesBuffer[stateIdx]->endOfStorage(), Memory::DefaultAlignment);
    }

    // MDDs
    for (unsigned int mddIdx = 0; mddIdx < mdds.getCapacity(); mddIdx += 1)
    {
        new (mdds[mddIdx]) DD::MDD<ProblemType, StateType>(problem, width, mallocType);
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
    setTop(index);
    barrier();
    mdds[index]->buildTopDown(DD::Type::Restricted, neighbourhoods[index]);
    barrier();
    setUpperbound(index);
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType, StateType>::clear()
{
    size = 0;
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType, StateType>::enqueue(BB::AugmentedState<StateType> const* augmentedState)
{
    unsigned int const index = size;
    *statesBuffer[index] = *augmentedState->state;
    augmentedStates[index]->state = statesBuffer[index];
    size += 1;
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType, StateType>::generateNeighbourhoods(StateType const* currentSolution, unsigned int eqPercentage, unsigned int neqPercentage, Array<std::mt19937>* rngs, Vector<std::thread>* cpuThreads)
{
    cpuThreads->clear();
    unsigned int step = cpuThreads->getCapacity();
    unsigned int end = size;
    for (unsigned int begin = 0; begin < cpuThreads->getCapacity(); begin += 1)
    {
        cpuThreads->resize(cpuThreads->getSize() + 1);
        new (cpuThreads->back()) std::thread(&OffloadBuffer<ProblemType, StateType>::generateNeighbourhoodsLoop, this, currentSolution, eqPercentage, neqPercentage, rngs, begin, step, end);
    }

}

template<typename ProblemType, typename StateType>
BB::AugmentedState<StateType> const* OffloadBuffer<ProblemType, StateType>::getAugmentedState(unsigned int index) const
{
    return augmentedStates[index];
}augmentedStates

template<typename ProblemType, typename StateType>
DD::MDD<ProblemType, StateType> const* OffloadBuffer<ProblemType, StateType>::getMDD(unsigned int index) const
{
    return mdds[index];
}
augmentedStates
template<typename ProblemType, typename StateType>
unsigned int OffloadBuffer<ProblemType, StateType>::getSize() const
{
    return size;
}

template<typename ProblemType, typename StateType>
bool OffloadBuffer<ProblemType, StateType>::isEmpty() const
{
    return size == 0;
}

template<typename ProblemType, typename StateType>
bool OffloadBuffer<ProblemType, StateType>::isFull() const
{
    return size == statesBuffer.getCapacity();
}


template<typename ProblemType, typename StateType>
__host__ __device__
void OffloadBuffer<ProblemType, StateType>::setTop(unsigned int index)
{
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0)
#endif
    {
        mdds[index]->top = *augmentedStates[index]->state;
    }
}
template<typename ProblemType, typename StateType>
__host__ __device__ void OffloadBuffer<ProblemType, StateType>::setUpperbound(unsigned int index)
{
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0)
#endif
    {
        augmentedStates[index]->upperbound = mdds[index]->bottom.cost;
    }
}

template<typename ProblemType, typename StateType>
void OffloadBuffer<ProblemType, StateType>::generateNeighbourhoodsLoop(StateType const* currentSolution, unsigned int eqPercentage, unsigned int neqPercentage, Array<std::mt19937>* rngs, unsigned int begin, unsigned int step, unsigned int end)
{
    for(unsigned int index = begin; index < end; index += step)
    {
        neighbourhoods[index]->generate(&currentSolution->selectedValues, eqPercentage, neqPercentage, rngs->at(index));
    }
}
