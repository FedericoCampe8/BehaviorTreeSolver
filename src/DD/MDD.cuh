#pragma once

#include <algorithm>
#include <thrust/binary_search.h>
#include <thrust/partition.h>
#include <thrust/functional.h>
#include <thrust/equal.h>
#include <Containers/Vector.cuh>
#include <Utils/Algorithms.cuh>
#include <Utils/CUDA.cuh>
#include "../DP/TSPPDState.cuh"
#include "../DP/TSPPDModel.cuh"
#include "../OP/Problem.cuh"
#include <LNS/Neighbourhood.cuh>
#include "Context.h"
#include "StateMetadata.cuh"

namespace DD
{
    template<typename ProblemType, typename StateType>
    class MDD
    {
        // Members
        public:
        u32 const width;
        ProblemType const * const problem;
        Vector<StateType> cutsetStates;
        private:
        std::byte* scratchpadMemory;
        LightVector<StateType>* currentStates;
        LightVector<StateType>* nextStates;
        LightVector<StateMetadata>* nextStatesMetadata;

        // Functions
        public:
        MDD(ProblemType const * problem, u32 width, Memory::MallocType mallocType);
        __host__ __device__ inline void buildTopDown(Neighbourhood const * neighbourhood, StateType const * top, StateType * bottom, bool lns);
        static u32 sizeOfScratchpadMemory(ProblemType const * problem, u32 width);
        private:
        __host__ __device__ inline void calcNextStatesMetadata(u32 variableIdx, Neighbourhood const * neighbourhood, bool lns);
        __host__ __device__ inline void initializeNextStatesMetadata();
        __host__ __device__ inline void finalizeNextStatesMetadata();
        __host__ __device__ inline void calcNextStates(u32 variableIdx);
        __host__ __device__ inline void initializeScratchpadMemory();
        __host__ __device__ inline void initializeTop(StateType const * top);
        __host__ __device__ inline void saveCutsetStates();
        __host__ __device__ void saveInvalidBottom(StateType * bottom);
        __host__ __device__ void saveBottom(StateType * bottom);
        __host__ __device__ void swapCurrentAndNextStates();
    };
}

template<typename ProblemType, typename StateType>
DD::MDD<ProblemType, StateType>::MDD(ProblemType const * problem, u32 width, Memory::MallocType mallocType) :
    width(width),
    problem(problem),
    cutsetStates(width, mallocType),
    scratchpadMemory(Memory::safeMalloc(sizeOfScratchpadMemory(problem, width), Memory::MallocType::Std)),
    currentStates(nullptr),
    nextStates(nullptr),
    nextStatesMetadata(nullptr)
{
    // Cutset states
    std::byte* storage = StateType::mallocStorages(problem, cutsetStates.getCapacity(), mallocType);
    for (u32 cutsetStateIdx = 0; cutsetStateIdx < cutsetStates.getCapacity(); cutsetStateIdx += 1)
    {
        StateType* cutsetState = cutsetStates.LightArray<StateType>::at(cutsetStateIdx);
        new (cutsetState) StateType(problem, storage);
        storage = Memory::align(cutsetState->endOfStorage(), Memory::DefaultAlignment);
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::buildTopDown(Neighbourhood const* neighbourhood, StateType const * top, StateType * bottom, bool lns)
{
    initializeScratchpadMemory();
    initializeTop(top);
    bool cutsetStatesSaved = false;
    u32 const variableIdxBegin = currentStates->at(0)->selectedValues.getSize();
    u32 const variableIdxEnd = problem->variables.getCapacity();
    for (u32 variableIdx = variableIdxBegin; variableIdx < variableIdxEnd; variableIdx += 1)
    {
        calcNextStatesMetadata(variableIdx, neighbourhood, lns);
        if (nextStatesMetadata->isEmpty())
        {
            saveInvalidBottom(bottom);
            return;
        }
        calcNextStates(variableIdx);
        if (not (lns or cutsetStatesSaved))
        {
            saveCutsetStates();
            cutsetStatesSaved = true;
        }
        swapCurrentAndNextStates();
    }
    saveBottom(bottom);

}

template<typename ProblemType, typename StateType>
u32 DD::MDD<ProblemType, StateType>::sizeOfScratchpadMemory(ProblemType const * problem, u32 width)
{
    u32 const sizeCurrentStates = sizeof(LightVector<StateType>) + LightVector<StateType>::sizeOfStorage(width) + (StateType::sizeOfStorage(problem) * width) + Memory::DefaultAlignmentPadding * 3;
    u32 const sizeNextStates =    sizeof(LightVector<StateType>) + LightVector<StateType>::sizeOfStorage(width) + (StateType::sizeOfStorage(problem) * width) + Memory::DefaultAlignmentPadding * 3;
    u32 const sizeNextStatesMetadata = sizeof(LightVector<StateMetadata>) + LightVector<StateMetadata>::sizeOfStorage(width * problem->maxBranchingFactor) + Memory::DefaultAlignmentPadding * 2;
     u32 const size = sizeCurrentStates + sizeNextStates + sizeNextStatesMetadata;
    return size + Memory::DefaultAlignmentPadding;
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::calcNextStatesMetadata(u32 variableIdx, Neighbourhood const * neighbourhood, bool lns)
{
    initializeNextStatesMetadata();

    u32 const elements = currentStates->getSize() * problem->maxBranchingFactor;
#ifdef __CUDA_ARCH__
    u32 const threads = blockDim.x;
    for(u32 index = threadIdx.x; index < elements; index += threads)
#else
    for (u32 index = 0; index < elements; index += 1)
#endif
    {
        u32 const currentStateIdx = index / problem->maxBranchingFactor;
        StateType const * const currentState = currentStates->at(currentStateIdx);
        OP::ValueType value = index % problem->maxBranchingFactor;
        if(currentState->admissibleValuesMap.contains(value))
        {
            if((not lns) or neighbourhood->constraintsCheck(variableIdx, value))
            {
                u32 const stateMetadataIdx = problem->maxBranchingFactor * currentStateIdx + value;
                nextStatesMetadata->at(stateMetadataIdx)->cost = calcCost(problem, currentState, value);
            }
        }
    }
    CUDA_BLOCK_BARRIER

    finalizeNextStatesMetadata();
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::calcNextStates(u32 variableIdx)
{
    //Resize
    CUDA_ONLY_FIRST_THREAD
    {
        u32 const nextStatesMetadataSize = nextStatesMetadata->getSize();
        thrust::minimum min;
        nextStates->resize(min(width, nextStatesMetadataSize));
    }
    CUDA_BLOCK_BARRIER

    
    u32 const elements = nextStates->getSize();
#ifdef __CUDA_ARCH__
    u32 const threads = blockDim.x;
    for(u32 nextStateIdx = threadIdx.x; nextStateIdx < elements; nextStateIdx += threads)
#else
    for (u32 nextStateIdx = 0; nextStateIdx < elements; nextStateIdx += 1)
#endif
    {
        StateMetadata const sm = *nextStatesMetadata->at(nextStateIdx);
        u32 const currentStateIdx = sm.index / problem->maxBranchingFactor;
        OP::ValueType const value = sm.index % problem->maxBranchingFactor;
        makeState(problem, currentStates->at(currentStateIdx), value, nextStatesMetadata->at(nextStateIdx)->cost, nextStates->at(nextStateIdx));
    }
    CUDA_BLOCK_BARRIER
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::initializeScratchpadMemory()
{
    CUDA_ONLY_FIRST_THREAD
    {
        std::byte* freeScratchpadMemory = nullptr;
#ifdef __CUDA_ARCH__
        extern __shared__ u32 sharedMemory[];
        freeScratchpadMemory = reinterpret_cast<std::byte*>(&sharedMemory);
#else
        freeScratchpadMemory = scratchpadMemory;
#endif
        // Current states
        std::byte const * const freeScratchpadMemoryBeforeCurrentStates = freeScratchpadMemory;
        currentStates = reinterpret_cast<LightVector<StateType>*>(freeScratchpadMemory);
        freeScratchpadMemory += sizeof(LightVector<StateType>);
        freeScratchpadMemory = Memory::align(freeScratchpadMemory, Memory::DefaultAlignment);
        new (currentStates) LightVector<StateType>(width, reinterpret_cast<StateType*>(freeScratchpadMemory));
        freeScratchpadMemory = Memory::align(currentStates->endOfStorage(), Memory::DefaultAlignment);
        for (u32 currentStateIdx = 0; currentStateIdx < currentStates->getCapacity(); currentStateIdx += 1)
        {
            StateType* currentState = currentStates->LightArray<StateType>::at(currentStateIdx);
            new (currentState) StateType(problem, freeScratchpadMemory);
            freeScratchpadMemory = Memory::align(currentState->endOfStorage(), Memory::DefaultAlignment);
        }
        u32 const sizeCurrentStates = reinterpret_cast<uintptr_t>(freeScratchpadMemory) - reinterpret_cast<uintptr_t>(freeScratchpadMemoryBeforeCurrentStates);


        // Next states
        std::byte const * const freeScratchpadMemoryBeforeNextStates = freeScratchpadMemory;
        nextStates = reinterpret_cast<LightVector<StateType>*>(freeScratchpadMemory);
        freeScratchpadMemory += sizeof(LightVector<StateType>);
        freeScratchpadMemory = Memory::align(freeScratchpadMemory, Memory::DefaultAlignment);
        new (nextStates) LightVector<StateType>(width, reinterpret_cast<StateType*>(freeScratchpadMemory));
        freeScratchpadMemory = Memory::align(nextStates->endOfStorage(), Memory::DefaultAlignment);
        for (u32 nextStateIdx = 0; nextStateIdx < nextStates->getCapacity(); nextStateIdx += 1)
        {
            StateType* nextState = nextStates->LightArray<StateType>::at(nextStateIdx);
            new (nextState) StateType(problem, freeScratchpadMemory);
            freeScratchpadMemory = Memory::align(nextState->endOfStorage(), Memory::DefaultAlignment);
        }
        u32 const sizeNextStates = reinterpret_cast<uintptr_t>(freeScratchpadMemory) - reinterpret_cast<uintptr_t>(freeScratchpadMemoryBeforeNextStates);


        // Next states metadata
        std::byte* const freeScratchpadMemoryBeforeNextStatesMetadata = freeScratchpadMemory;
        nextStatesMetadata = reinterpret_cast<LightVector<StateMetadata>*>(freeScratchpadMemory);
        freeScratchpadMemory += sizeof(LightVector<StateMetadata>);
        freeScratchpadMemory = Memory::align(freeScratchpadMemory, Memory::DefaultAlignment);
        new (nextStatesMetadata) LightVector<StateMetadata>(width * problem->maxBranchingFactor, reinterpret_cast<StateMetadata*>(freeScratchpadMemory));
        freeScratchpadMemory = nextStatesMetadata->endOfStorage();
        u32 const sizeNextStatesMetadata = reinterpret_cast<uintptr_t>(freeScratchpadMemory) - reinterpret_cast<uintptr_t>(freeScratchpadMemoryBeforeNextStatesMetadata);

        // Memory
        [[maybe_unused]] u32 const usedScratchpadMemory = sizeCurrentStates + sizeNextStates + sizeNextStatesMetadata;
    }
    CUDA_BLOCK_BARRIER
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::initializeTop(StateType const * top)
{
    CUDA_ONLY_FIRST_THREAD
    {
        currentStates->resize(1);
        *currentStates->at(0) = *top;
    }
    CUDA_BLOCK_BARRIER
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::saveCutsetStates()
{
    //Resize
    CUDA_ONLY_FIRST_THREAD
    {
        thrust::minimum min;
        cutsetStates.resize(min(width,nextStatesMetadata->getSize()));
    }
    CUDA_BLOCK_BARRIER

    u32 const elements = cutsetStates.getSize();
#ifdef __CUDA_ARCH__
    u32 const threads = blockDim.x;
    for(u32 cutsetStateIdx = threadIdx.x; cutsetStateIdx < elements; cutsetStateIdx += threads)
#else
    for (u32 cutsetStateIdx = 0; cutsetStateIdx < cutsetStates.getSize(); cutsetStateIdx += 1)
#endif
    {
        *cutsetStates[cutsetStateIdx] = *nextStates->at(cutsetStateIdx);
    }
    CUDA_BLOCK_BARRIER
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::swapCurrentAndNextStates()
{
    CUDA_ONLY_FIRST_THREAD
    {
        LightVector<StateType>::swap(*currentStates, *nextStates);
    }
    CUDA_BLOCK_BARRIER
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::initializeNextStatesMetadata()
{
    //Resize
    u32 elements = currentStates->getSize() * problem->maxBranchingFactor;
    CUDA_ONLY_FIRST_THREAD
    {
       nextStatesMetadata->resize(elements);
    }
    CUDA_BLOCK_BARRIER

    //Reset
#ifdef __CUDA_ARCH__
u32 const threads = blockDim.x;
    for(u32 stateMetadataIdx = threadIdx.x; stateMetadataIdx < elements; stateMetadataIdx += threads)
#else
    for (u32 stateMetadataIdx = 0; stateMetadataIdx < elements; stateMetadataIdx += 1)
#endif
    {
        nextStatesMetadata->at(stateMetadataIdx)->cost = DP::MaxCost;
        nextStatesMetadata->at(stateMetadataIdx)->index = stateMetadataIdx;
    }
    CUDA_BLOCK_BARRIER
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::finalizeNextStatesMetadata()
{
    Algorithms::sort(nextStatesMetadata->begin(), nextStatesMetadata->getSize());

    CUDA_ONLY_FIRST_THREAD
    {
        StateMetadata const invalidStateMetadata(DP::MaxCost, 0);
        StateMetadata const * const nextStatesMetadataEnd = thrust::lower_bound(thrust::seq, nextStatesMetadata->begin(), nextStatesMetadata->end(), invalidStateMetadata);
        nextStatesMetadata->resize(nextStatesMetadataEnd);
    }
    CUDA_BLOCK_BARRIER
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::saveInvalidBottom(StateType * bottom)
{
    CUDA_ONLY_FIRST_THREAD
    {
        bottom->invalidate();
    }
    CUDA_BLOCK_BARRIER
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::saveBottom(StateType* bottom)
{
    CUDA_ONLY_FIRST_THREAD
    {
        *bottom = *currentStates->at(0);
        //printf("[DEBUG] Bottom cost: %d\n", currentStates->at(0)->cost);
    }
    CUDA_BLOCK_BARRIER
}

