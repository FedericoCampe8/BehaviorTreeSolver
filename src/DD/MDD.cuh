#pragma once

#include <algorithm>
#include <thrust/binary_search.h>
#include <thrust/partition.h>
#include <thrust/functional.h>
#include <thrust/equal.h>
#include <Containers/Vector.cuh>
#include <Utils/Algorithms.cuh>
#include <Utils/CUDA.cuh>
#include <Utils/Random.cuh>
#include <OP/Problem.cuh>
#include <LNS/Neighbourhood.cuh>
#include <DD/Context.h>
#include <DD/StateMetadata.cuh>

namespace DD
{
    template<typename ProblemType, typename StateType>
    class MDD
    {
        // Members
        private:
        u32 const width;
        ProblemType const * const problem;
        std::byte* scratchpadMemory;
        LightVector<StateType>* currentStates;
        LightVector<StateType>* nextStates;
        LightVector<StateMetadata>* nextStatesMetadata;

        // Functions
        public:
        MDD(ProblemType const * problem, u32 width);
        __host__ __device__ inline void buildTopDown(StateType const * top, StateType * bottom, Vector<StateType> * cutset, Neighbourhood const * neighbourhood, RandomEngine* randomEngine, bool lns);
        static u32 sizeOfScratchpadMemory(ProblemType const * problem, u32 width);
        private:
        __host__ __device__ inline void calcNextStatesMetadata(u32 variableIdx, Neighbourhood const * neighbourhood, RandomEngine * randomEngine, bool lns);
        __host__ __device__ inline void initializeNextStatesMetadata(RandomEngine * randomEngine);
        __host__ __device__ inline void finalizeNextStatesMetadata();
        __host__ __device__ inline void calcNextStates(u32 variableIdx);
        __host__ __device__ inline void initializeScratchpadMemory();
        __host__ __device__ inline void initializeTop(StateType const * top);
        __host__ __device__ inline void saveCutset(Vector<StateType> * cutset);
        __host__ __device__ void saveInvalidBottom(StateType * bottom);
        __host__ __device__ void saveBottom(StateType * bottom);
        __host__ __device__ void swapCurrentAndNextStates();
    };
}

template<typename ProblemType, typename StateType>
DD::MDD<ProblemType, StateType>::MDD(ProblemType const * problem, u32 width) :
    width(width),
    problem(problem),
    scratchpadMemory(Memory::safeMalloc(sizeOfScratchpadMemory(problem, width), Memory::MallocType::Std)),
    currentStates(nullptr),
    nextStates(nullptr),
    nextStatesMetadata(nullptr)
{}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::buildTopDown(StateType const * top, StateType * bottom, Vector<StateType> * cutset, Neighbourhood const * neighbourhood, RandomEngine * randomEngine, bool lns)
{
    initializeScratchpadMemory();
    initializeTop(top);
    bool cutsetStatesSaved = false;
    u32 const variableIdxBegin = currentStates->at(0)->selectedValues.getSize();
    u32 const variableIdxEnd = problem->variables.getCapacity();
    for (u32 variableIdx = variableIdxBegin; variableIdx < variableIdxEnd; variableIdx += 1)
    {
        calcNextStatesMetadata(variableIdx, neighbourhood, randomEngine, lns);
        if (nextStatesMetadata->isEmpty())
        {
            saveInvalidBottom(bottom);
            return;
        }
        calcNextStates(variableIdx);
        if (not (lns or cutsetStatesSaved))
        {
            saveCutset(cutset);
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
    u32 const sizeNextStates = sizeof(LightVector<StateType>) + LightVector<StateType>::sizeOfStorage(width) + (StateType::sizeOfStorage(problem) * width) + Memory::DefaultAlignmentPadding * 3;
    u32 const sizeNextStatesMetadata = sizeof(LightVector<StateMetadata>) + LightVector<StateMetadata>::sizeOfStorage(width * problem->maxBranchingFactor) + Memory::DefaultAlignmentPadding * 2;
    u32 const size = sizeCurrentStates + sizeNextStates + sizeNextStatesMetadata;
    return size + Memory::DefaultAlignmentPadding;
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::calcNextStatesMetadata(u32 variableIdx, Neighbourhood const * neighbourhood, RandomEngine * randomEngine, bool lns)
{
    initializeNextStatesMetadata(randomEngine);

    u32 const elements = currentStates->getSize() * problem->maxBranchingFactor;
#ifdef __CUDA_ARCH__
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(threadIdx.x, blockDim.x, elements);
#else
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(0, 1, elements);
#endif
    for (u32 index = indicesBeginEnd.first; index < indicesBeginEnd.second; index += 1)
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
        nextStates->resize(min(width, nextStatesMetadataSize));
    }
    CUDA_BLOCK_BARRIER

    u32 const elements = nextStates->getSize();
#ifdef __CUDA_ARCH__
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(threadIdx.x, blockDim.x, elements);
#else
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(0, 1, elements);
#endif
    for (u32 nextStateIdx = indicesBeginEnd.first; nextStateIdx < indicesBeginEnd.second; nextStateIdx += 1)

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
        [[maybe_unused]]
        u32 const usedScratchpadMemory = sizeCurrentStates + sizeNextStates + sizeNextStatesMetadata;
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
void DD::MDD<ProblemType, StateType>::saveCutset(Vector<StateType> * cutset)
{
    //Resize
    CUDA_ONLY_FIRST_THREAD
    {
        cutset->resize(min(width,nextStatesMetadata->getSize()));
    }
    CUDA_BLOCK_BARRIER

    u32 const elements = cutset->getSize();

#ifdef __CUDA_ARCH__
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(threadIdx.x, blockDim.x, elements);
#else
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(0, 1, elements);
#endif
    for (u32 stateIdx = indicesBeginEnd.first; stateIdx < indicesBeginEnd.second; stateIdx += 1)
    {
        *cutset->at(stateIdx) = *nextStates->at(stateIdx);
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
void DD::MDD<ProblemType, StateType>::initializeNextStatesMetadata(RandomEngine* randomEngine)
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
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(threadIdx.x, blockDim.x, elements);
#else
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(0, 1, elements);
#endif
    for (u32 stateMetadataIdx = indicesBeginEnd.first; stateMetadataIdx < indicesBeginEnd.second; stateMetadataIdx += 1)
    {
        nextStatesMetadata->at(stateMetadataIdx)->cost = DP::MaxCost;
        nextStatesMetadata->at(stateMetadataIdx)->index = stateMetadataIdx;
        nextStatesMetadata->at(stateMetadataIdx)->random = randomEngine->getFloat01();
    }
    CUDA_BLOCK_BARRIER
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::finalizeNextStatesMetadata()
{
#ifdef __CUDA_ARCH__
    Algorithms::sort(nextStatesMetadata->begin(), nextStatesMetadata->end());
#else
    std::sort(nextStatesMetadata->begin(), nextStatesMetadata->end());
#endif

    CUDA_ONLY_FIRST_THREAD
    {
        StateMetadata const * const nextStatesMetadataEnd = thrust::partition_point(thrust::seq, nextStatesMetadata->begin(), nextStatesMetadata->end(), &StateMetadata::isValid);
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
    }
    CUDA_BLOCK_BARRIER
}

