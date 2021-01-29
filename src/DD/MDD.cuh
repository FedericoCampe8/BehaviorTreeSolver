#pragma once

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <Containers/Vector.cuh>
#include <Utils/Algorithms.cuh>
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
        unsigned int const width;
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
        MDD(ProblemType const * problem, unsigned int width, Memory::MallocType mallocType);
        __host__ __device__ inline void buildTopDown(DD::Type type, Neighbourhood const * neighbourhood);
        __host__ __device__ unsigned int sizeOfScratchpadMemory() const;
        private:
        __host__ __device__ inline void barrier() const;
        __host__ __device__ inline void calcAuxiliaryData(unsigned int variableIdx, Neighbourhood const * neighbourhood);
        __host__ __device__ inline void calcNextStates(unsigned int variableIdx);
        __host__ __device__ inline void initScratchpadMemory();
        __host__ __device__ inline void initTop();
        __host__ __device__ void mergeNextStates(unsigned int variableIdx);
        __host__ __device__ inline void resetAuxiliaryData();
        __host__ __device__ inline void resizeAuxiliaryData();
        __host__ __device__ inline void resizeCutset();
        __host__ __device__ void resizeNextStates(unsigned int variableIdx, unsigned int variablesCount);
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
    cutset(width * problem->maxBranchingFactor, mallocType),
    scratchpadMemory(Memory::safeMalloc(sizeOfScratchpadMemory(), Memory::MallocType::Std))
{
    // Cutset states
    unsigned int storageSize = StateType::sizeOfStorage(problem);
    std::byte* storages = StateType::mallocStorages(problem, cutset.getCapacity(), mallocType);
    for (unsigned int cutsetStateIdx = 0; cutsetStateIdx < cutset.getCapacity(); cutsetStateIdx += 1)
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
    barrier();
    bool cutsetSaved = false;
    unsigned int const variablesCount = problem->variables.getCapacity();
    for (unsigned int variableIdx = currentStates->at(0)->selectedValues.getSize(); variableIdx < variablesCount; variableIdx += 1)
    {
        resizeAuxiliaryData();
        barrier();
        resetAuxiliaryData();
        barrier();
        calcAuxiliaryData(variableIdx, neighbourhood);
        barrier();
        sortAuxiliaryData();
        barrier();
        shirkToFitAuxiliaryData();
        resizeNextStates(variableIdx, variablesCount);
        barrier();
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
            barrier();
            saveCutset();
            cutsetSaved = true;
        }
        barrier();
        swapCurrentAndNextStates();
    }
    saveBottom();
}

template<typename ProblemType, typename StateType>
__host__ __device__
unsigned int DD::MDD<ProblemType, StateType>::sizeOfScratchpadMemory() const
{
    return
        sizeof(LightVector<StateType>) +
        LightVector<StateType>::sizeOfStorage(width) +
        StateType::sizeOfStorage(problem) * width +  // currentStates
        sizeof(LightVector<StateType>) +
        LightVector<StateType>::sizeOfStorage(width) +
        StateType::sizeOfStorage(problem) * width +  // nextStates
        sizeof(LightVector<AuxiliaryData>) +
        LightVector<AuxiliaryData>::sizeOfStorage(width * problem->maxBranchingFactor) +
        Memory::AlignmentPadding * 7; // alignment
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::barrier() const
{
#ifdef __CUDA_ARCH__
    __syncthreads();
#endif
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::calcAuxiliaryData(unsigned int variableIdx, Neighbourhood const* neighbourhood)
{
#ifdef __CUDA_ARCH__
    unsigned int const currentStateIdx = threadIdx.x / problem->maxBranchingFactor;
    if (currentStateIdx < currentStates->getSize())
#else
        for (unsigned int currentStateIdx = 0; currentStateIdx < currentStates->getSize(); currentStateIdx += 1)
#endif
    {
        StateType const* const currentState = currentStates->at(currentStateIdx);
#ifdef __CUDA_ARCH__
        unsigned int const admissibleValueIdx = threadIdx.x % problem->maxBranchingFactor;
        if (admissibleValueIdx < currentState->admissibleValues.getSize())
#else
            for (unsigned int admissibleValueIdx = 0; admissibleValueIdx < currentState->admissibleValues.getSize(); admissibleValueIdx += 1)
#endif
        {
            OP::ValueType const value = * currentState->admissibleValues[admissibleValueIdx];
            bool boundsChk = problem->variables[variableIdx]->boundsCheck(value);
            bool constraintsChk = neighbourhood->constraintsCheck(variableIdx, value);
            if (boundsChk and constraintsChk)
            {
                unsigned int const valueIdx = value - problem->variables[variableIdx]->minValue;
                unsigned int const auxiliaryDataIdx = problem->maxBranchingFactor * currentStateIdx + valueIdx;
                auxiliaryData->at(auxiliaryDataIdx)->cost = calcCost(problem, currentState, value);
            }
        }
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::calcNextStates(unsigned int variableIdx)
{
#ifdef __CUDA_ARCH__
    unsigned int const nextStateIdx = threadIdx.x;
    if (nextStateIdx < nextStates->getSize())
#else
        for (unsigned int nextStateIdx = 0; nextStateIdx < nextStates->getSize(); nextStateIdx += 1)
#endif
    {
        AuxiliaryData ad = *auxiliaryData->at(nextStateIdx);
        unsigned int const currentStateIdx = ad.index / problem->maxBranchingFactor;
        unsigned int const valueIdx = ad.index % problem->maxBranchingFactor;
        OP::ValueType const value = problem->variables[variableIdx]->minValue + valueIdx;
        makeState(problem, currentStates->at(currentStateIdx), value, auxiliaryData->at(nextStateIdx)->cost, nextStates->at(nextStateIdx));
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::initScratchpadMemory()
{
    std::byte* freeScratchpadMemory;
#ifdef __CUDA_ARCH__
    extern __shared__ uint32_t sharedMemory[];
    freeScratchpadMemory = reinterpret_cast<std::byte*>(& sharedMemory);
#else
    freeScratchpadMemory = scratchpadMemory;
#endif
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0)
#endif
    {
        // Current states
        currentStates = reinterpret_cast<LightVector<StateType>*>(freeScratchpadMemory);
        freeScratchpadMemory += sizeof(LightVector<StateType>);
        freeScratchpadMemory = Memory::align(8, freeScratchpadMemory);
        new (currentStates) LightVector<StateType>(width, reinterpret_cast<StateType*>(freeScratchpadMemory));
        freeScratchpadMemory = Memory::align(8, currentStates->endOfStorage());
        unsigned int const storageSize = StateType::sizeOfStorage(problem);
        for (unsigned int stateIdx = 0; stateIdx < currentStates->getCapacity(); stateIdx += 1)
        {
            new (currentStates->LightArray<StateType>::at(stateIdx)) StateType(problem, freeScratchpadMemory);
            freeScratchpadMemory += storageSize;
        }
        freeScratchpadMemory = Memory::align(8, freeScratchpadMemory);

        // Next states
        nextStates = reinterpret_cast<LightVector<StateType>*>(freeScratchpadMemory);
        freeScratchpadMemory += sizeof(LightVector<StateType>);
        freeScratchpadMemory = Memory::align(8, freeScratchpadMemory);
        new(nextStates) LightVector<StateType>(width, reinterpret_cast<StateType*>(freeScratchpadMemory));
        freeScratchpadMemory = Memory::align(8, nextStates->endOfStorage());
        for (unsigned int stateIdx = 0; stateIdx < nextStates->getCapacity(); stateIdx += 1)
        {
            new (nextStates->LightArray<StateType>::at(stateIdx)) StateType(problem, freeScratchpadMemory);
            freeScratchpadMemory += storageSize;
        }
        freeScratchpadMemory = Memory::align(8, freeScratchpadMemory);

        // Auxiliary information
        auxiliaryData = reinterpret_cast<LightVector<AuxiliaryData>*>(freeScratchpadMemory);
        freeScratchpadMemory += sizeof(LightVector<AuxiliaryData>);
        freeScratchpadMemory = Memory::align(8, freeScratchpadMemory);
        new (auxiliaryData) LightVector<AuxiliaryData>(width * problem->maxBranchingFactor, reinterpret_cast<AuxiliaryData*>(freeScratchpadMemory));
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::initTop()
{
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0)
#endif
    {
        currentStates->resize(1);
        *currentStates->at(0) = top;
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::mergeNextStates(unsigned int variableIdx)
{
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0)
#endif
    {
        for (unsigned int auxiliaryDataIdx = nextStates->getSize(); auxiliaryDataIdx < auxiliaryData->getSize(); auxiliaryDataIdx += 1)
        {
            AuxiliaryData ad = *auxiliaryData->at(auxiliaryDataIdx);
            unsigned int const currentStateIdx = ad.index / problem->maxBranchingFactor;
            unsigned int const valueIdx = ad.index % problem->maxBranchingFactor;
            OP::ValueType const value = problem->variables[variableIdx]->minValue + valueIdx;
            mergeState(problem, currentStates->at(currentStateIdx), value, nextStates->back());
        }
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::resetAuxiliaryData()
{
#ifdef __CUDA_ARCH__
    unsigned int const auxiliaryDataIdx = threadIdx.x;
    if (auxiliaryDataIdx < auxiliaryData->getSize())
#else
    for (unsigned int auxiliaryDataIdx = 0; auxiliaryDataIdx < auxiliaryData->getSize(); auxiliaryDataIdx += 1)
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
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0)
#endif
    {
        auxiliaryData->resize(currentStates->getSize() * problem->maxBranchingFactor);
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::resizeCutset()
{
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0)
#endif
    {
        cutset.resize(nextStates->getSize());
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::resizeNextStates(unsigned int variableIdx, unsigned int variablesCount)
{
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0)
#endif
    {
        unsigned int const nextStatesCount = variableIdx < variablesCount - 1 ? width : 1;
        nextStates->resize(min(nextStatesCount, auxiliaryData->getSize()));
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::saveCutset()
{
#ifdef __CUDA_ARCH__
    unsigned int const cutsetStateIdx = threadIdx.x;
    if (cutsetStateIdx < cutset.getSize())
#else
    for (unsigned int cutsetStateIdx = 0; cutsetStateIdx < cutset.getSize(); cutsetStateIdx += 1)
#endif
    {
        *cutset[cutsetStateIdx] = *nextStates->at(cutsetStateIdx);
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__ void DD::MDD<ProblemType, StateType>::saveBottom()
{
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0)
#endif
    {
        bottom = *currentStates->at(0);
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::setInvalid()
{
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0)
#endif
    {
        bottom.setInvalid();
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::shirkToFitAuxiliaryData()
{
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0)
#endif
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
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0)
#endif
    {
        LightVector<StateType>::swap(*currentStates, *nextStates);
    }
}