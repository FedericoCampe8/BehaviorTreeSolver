#pragma once

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <Containers/Vector.cuh>
#include <Utils/Algorithms.cuh>
#include <Utils/CUDA.cuh>
#include "../DP/VRPState.cuh"
#include "../DP/VRPModel.cuh"
#include "../OP/Problem.cuh"
#include "../Neighbourhood.cuh"
#include "Context.h"
#include "AuxiliaryData.cuh"

namespace DD
{
    template<typename ProblemType, typename StateType>
    class MDD
    {
        // Members
        public:
        u32 const width;
        ProblemType const * const problem;
        StateType top;
        StateType bottom;
        Vector<StateType> cutset;
        private:
        std::byte* scratchpadMemory;
        LightVector<StateType>* currentStates;
        LightVector<StateType>* nextStates;
        LightVector<AuxiliaryData>* auxiliaryData;

        // Functions
        public:
        MDD(ProblemType const * problem, u32 width, Memory::MallocType mallocType);
        __host__ __device__ inline void buildTopDown(DD::Type type, Neighbourhood const * neighbourhood);
        __host__ __device__ u32 sizeOfScratchpadMemory() const;
        private:
        __host__ __device__ inline void calcAuxiliaryData(u32 variableIdx, Neighbourhood const * neighbourhood);
        __host__ __device__ inline void calcNextStates(u32 variableIdx);
        __host__ __device__ inline void initScratchpadMemory();
        __host__ __device__ inline void initTop();
        __host__ __device__ void mergeNextStates(u32 variableIdx);
        __host__ __device__ inline void resetAuxiliaryData();
        __host__ __device__ inline void resizeAuxiliaryData();
        __host__ __device__ inline void resizeCutset();
        __host__ __device__ void resizeNextStates(u32 variableIdx, u32 variablesCount);
        __host__ __device__ inline void saveCutset();
        __host__ __device__ void saveBottom();
        __host__ __device__ void setInvalid();
        __host__ __device__ void shirkToFitAuxiliaryData();
        __host__ __device__ inline void sortAuxiliaryData();
        __host__ __device__ void swapCurrentAndNextStates();
    };
}

template<typename ProblemType, typename StateType>
DD::MDD<ProblemType, StateType>::MDD(ProblemType const * problem, unsigned int width, Memory::MallocType mallocType) :
    width(width),
    problem(problem),
    top(problem, mallocType),
    bottom(problem, mallocType),
    cutset(width * problem->maxValue, mallocType),
    scratchpadMemory(Memory::safeMalloc(sizeOfScratchpadMemory(), Memory::MallocType::Std))
{
    // Cutset states
    u32 const storageSize = StateType::sizeOfStorage(problem);
    std::byte* storages = StateType::mallocStorages(problem, cutset.getCapacity(), mallocType);
    for (u32 cutsetStateIdx = 0; cutsetStateIdx < cutset.getCapacity(); cutsetStateIdx += 1)
    {
        new (cutset.LightArray<StateType>::at(cutsetStateIdx)) StateType(problem, storages);
        storages += storageSize;
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::buildTopDown(DD::Type type, Neighbourhood const* neighbourhood)
{
    initScratchpadMemory();
    initTop();
    CUDA_THREADS_BARRIER
    bool cutsetSaved = false;
    u32 const variablesCount = problem->variables.getCapacity();
    for (u32 variableIdx = currentStates->at(0)->selectedValues.getSize(); variableIdx < variablesCount; variableIdx += 1)
    {
        resizeAuxiliaryData();
        CUDA_THREADS_BARRIER
        resetAuxiliaryData();
        CUDA_THREADS_BARRIER
        calcAuxiliaryData(variableIdx, neighbourhood);
        CUDA_THREADS_BARRIER
        sortAuxiliaryData();
        CUDA_THREADS_BARRIER
        shirkToFitAuxiliaryData();
        resizeNextStates(variableIdx, variablesCount);
        CUDA_THREADS_BARRIER
        if (nextStates->isEmpty())
        {
            setInvalid();
            return;
        }
        calcNextStates(variableIdx);
        if (type == Type::Relaxed)
        {
            mergeNextStates(variableIdx);
        }
        else if (not cutsetSaved)
        {
            resizeCutset();
            CUDA_THREADS_BARRIER
            saveCutset();
            cutsetSaved = true;
        }
        CUDA_THREADS_BARRIER
        swapCurrentAndNextStates();
    }
    saveBottom();
}

template<typename ProblemType, typename StateType>
__host__ __device__
u32 DD::MDD<ProblemType, StateType>::sizeOfScratchpadMemory() const
{
    return
        sizeof(LightVector<StateType>) + LightVector<StateType>::sizeOfStorage(width) + (StateType::sizeOfStorage(problem) * width) +  // currentStates
        sizeof(LightVector<StateType>) + LightVector<StateType>::sizeOfStorage(width) + (StateType::sizeOfStorage(problem) * width) +  // nextStates
        sizeof(LightVector<AuxiliaryData>) + LightVector<AuxiliaryData>::sizeOfStorage(width * problem->maxBranchingFactor) + // auxiliaryData
        Memory::DefaultAlignment * 7; // alignment padding
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::calcAuxiliaryData(u32 variableIdx, Neighbourhood const * neighbourhood)
{
#ifdef __CUDA_ARCH__
    u32 const currentStateIdx = threadIdx.x / problem->maxBranchingFactor;
    if (currentStateIdx < currentStates->getSize())
#else
    for (u32 currentStateIdx = 0; currentStateIdx < currentStates->getSize(); currentStateIdx += 1)
#endif
    {
        StateType const * const currentState = currentStates->at(currentStateIdx);
#ifdef __CUDA_ARCH__
        OP::ValueType const value = threadIdx.x % problem->maxBranchingFactor;
#else
        for (OP::ValueType value = 0; value < problem->maxBranchingFactor; value += 1)
#endif
        {
            if(currentState->admissibleValuesMap.contains(value))
            {
                bool const boundsChk = problem->variables[variableIdx]->boundsCheck(value);
                bool const constraintsChk = neighbourhood->constraintsCheck(variableIdx, value);
                if (boundsChk and constraintsChk)
                {
                    u32 const auxiliaryDataIdx = problem->maxBranchingFactor * currentStateIdx + value;
                    auxiliaryData->at(auxiliaryDataIdx)->cost = calcCost(problem, currentState, value);
                }
            }
        }
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::calcNextStates(unsigned int variableIdx)
{
#ifdef __CUDA_ARCH__
    u32 const nextStateIdx = threadIdx.x;
    if (nextStateIdx < nextStates->getSize())
#else
    for (u32 nextStateIdx = 0; nextStateIdx < nextStates->getSize(); nextStateIdx += 1)
#endif
    {
        AuxiliaryData ad = *auxiliaryData->at(nextStateIdx);
        u32 const currentStateIdx = ad.index / problem->maxBranchingFactor;
        OP::ValueType const value = ad.index % problem->maxBranchingFactor;
        makeState(problem, currentStates->at(currentStateIdx), value, auxiliaryData->at(nextStateIdx)->cost, nextStates->at(nextStateIdx));
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::initScratchpadMemory()
{
    std::byte* freeScratchpadMemory;
#ifdef __CUDA_ARCH__
    extern __shared__ u32 sharedMemory[];
    freeScratchpadMemory = reinterpret_cast<std::byte*>(&sharedMemory);
#else
    freeScratchpadMemory = scratchpadMemory;
#endif
    CUDA_ONLY_FIRST_THREAD
    {
        // Current states
        currentStates = reinterpret_cast<LightVector<StateType>*>(freeScratchpadMemory);
        freeScratchpadMemory += sizeof(LightVector<StateType>);
        freeScratchpadMemory = Memory::align(freeScratchpadMemory);
        new (currentStates) LightVector<StateType>(width, reinterpret_cast<StateType*>(freeScratchpadMemory));
        freeScratchpadMemory = Memory::align(currentStates->endOfStorage());
        u32 const storageSize = StateType::sizeOfStorage(problem);
        for (u32 stateIdx = 0; stateIdx < currentStates->getCapacity(); stateIdx += 1)
        {
            new (currentStates->LightArray<StateType>::at(stateIdx)) StateType(problem, freeScratchpadMemory);
            freeScratchpadMemory += storageSize;
        }
        freeScratchpadMemory = Memory::align(freeScratchpadMemory);

        // Next states
        nextStates = reinterpret_cast<LightVector<StateType>*>(freeScratchpadMemory);
        freeScratchpadMemory += sizeof(LightVector<StateType>);
        freeScratchpadMemory = Memory::align(freeScratchpadMemory);
        new(nextStates) LightVector<StateType>(width, reinterpret_cast<StateType*>(freeScratchpadMemory));
        freeScratchpadMemory = Memory::align(nextStates->endOfStorage());
        for (u32 stateIdx = 0; stateIdx < nextStates->getCapacity(); stateIdx += 1)
        {
            new (nextStates->LightArray<StateType>::at(stateIdx)) StateType(problem, freeScratchpadMemory);
            freeScratchpadMemory += storageSize;
        }
        freeScratchpadMemory = Memory::align(freeScratchpadMemory);

        // Auxiliary information
        auxiliaryData = reinterpret_cast<LightVector<AuxiliaryData>*>(freeScratchpadMemory);
        freeScratchpadMemory += sizeof(LightVector<AuxiliaryData>);
        freeScratchpadMemory = Memory::align(freeScratchpadMemory);
        new (auxiliaryData) LightVector<AuxiliaryData>(width * problem->maxBranchingFactor, reinterpret_cast<AuxiliaryData*>(freeScratchpadMemory));
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::initTop()
{
    CUDA_ONLY_FIRST_THREAD
    {
        currentStates->resize(1);
        *currentStates->at(0) = top;
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::mergeNextStates(unsigned int variableIdx)
{
    CUDA_ONLY_FIRST_THREAD
    {
        for (u32 auxiliaryDataIdx = nextStates->getSize(); auxiliaryDataIdx < auxiliaryData->getSize(); auxiliaryDataIdx += 1)
        {
            AuxiliaryData ad = *auxiliaryData->at(auxiliaryDataIdx);
            u32 const currentStateIdx = ad.index / problem->maxBranchingFactor;
            OP::ValueType const value = ad.index % problem->maxBranchingFactor;
            mergeState(problem, currentStates->at(currentStateIdx), value, nextStates->back());
        }
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::resetAuxiliaryData()
{
#ifdef __CUDA_ARCH__
    u32 const auxiliaryDataIdx = threadIdx.x;
    if (auxiliaryDataIdx < auxiliaryData->getSize())
#else
    for (u32 auxiliaryDataIdx = 0; auxiliaryDataIdx < auxiliaryData->getSize(); auxiliaryDataIdx += 1)
#endif
    {
        auxiliaryData->at(auxiliaryDataIdx)->cost = DP::MaxCost;
        auxiliaryData->at(auxiliaryDataIdx)->index = auxiliaryDataIdx;
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::resizeAuxiliaryData()
{
    CUDA_ONLY_FIRST_THREAD
    {
        auxiliaryData->resize(currentStates->getSize() * problem->maxBranchingFactor);
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::resizeCutset()
{
    CUDA_ONLY_FIRST_THREAD
    {
        cutset.resize(nextStates->getSize());
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::resizeNextStates(unsigned int variableIdx, unsigned int variablesCount)
{
    CUDA_ONLY_FIRST_THREAD
    {
        u32 const nextStatesMaxCount = variableIdx < variablesCount - 1 ? width : 1;
        nextStates->resize(min(nextStatesMaxCount, auxiliaryData->getSize()));
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::saveCutset()
{
#ifdef __CUDA_ARCH__
    u32 const cutsetStateIdx = threadIdx.x;
    if (cutsetStateIdx < cutset.getSize())
#else
    for (u32 cutsetStateIdx = 0; cutsetStateIdx < cutset.getSize(); cutsetStateIdx += 1)
#endif
    {
        *cutset[cutsetStateIdx] = *nextStates->at(cutsetStateIdx);
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::saveBottom()
{
    CUDA_ONLY_FIRST_THREAD
    {
        bottom = *currentStates->at(0);
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::setInvalid()
{
    CUDA_ONLY_FIRST_THREAD
    {
        bottom.makeInvalid();
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::shirkToFitAuxiliaryData()
{
    CUDA_ONLY_FIRST_THREAD
    {
        AuxiliaryData const invalidAuxiliaryData(DP::MaxCost, 0);
        AuxiliaryData const * const auxiliaryDataEnd = thrust::lower_bound(thrust::seq, auxiliaryData->begin(), auxiliaryData->end(), invalidAuxiliaryData);
        auxiliaryData->resize(auxiliaryDataEnd);
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::sortAuxiliaryData()
{
#ifdef __CUDA_ARCH__
    Algorithms::oddEvenSort(auxiliaryData->begin(), auxiliaryData->getSize());
#else
    thrust::sort(thrust::seq, auxiliaryData->begin(), auxiliaryData->end());
#endif
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::swapCurrentAndNextStates()
{
    CUDA_ONLY_FIRST_THREAD
    {
        LightVector<StateType>::swap(*currentStates, *nextStates);
    }
}