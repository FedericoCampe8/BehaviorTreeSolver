#pragma once

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/swap.h>
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
        private:
        Vector<StateType> cutset;
        Vector<StateType> currentStates;
        Vector<StateType> nextStates;
        Vector<AuxiliaryData> auxiliaryData;

        // Functions
        public:
        MDD(ProblemType const* problem, unsigned int width, Memory::MallocType mallocType);
        __host__ __device__ inline void buildTopDown(DD::Type type, Neighbourhood const * neighbourhood);
        __host__ __device__ Vector<StateType> const * getCutset() const;
        __host__ __device__ StateType* getBottom() const;
        __host__ __device__ void setTop(StateType const * state);
        __host__ __device__ void setInvalid();
        private:
        __host__ __device__ inline void calcAuxiliaryData(unsigned int variableIdx, Neighbourhood const * neighbourhood);
        __host__ __device__ inline void calcNextStates(unsigned int variableIdx);
        __host__ __device__ void mergeNextStates(unsigned int variableIdx);
        __host__ __device__ inline void resetAuxiliaryData();
        __host__ __device__ inline void resizeAuxiliaryData();
        __host__ __device__ inline void resizeCutset();
        __host__ __device__ void resizeNextStates(unsigned int variableIdx, unsigned int variablesCount);
        __host__ __device__ inline void saveCutset();
        __host__ __device__ void shirkToFitAuxiliaryData();
        __host__ __device__ inline void sortAuxiliaryData();
        __host__ __device__ void swapCurrentAndNextStates();

    };
}

template<typename ProblemType, typename StateType>
DD::MDD<ProblemType, StateType>::MDD(ProblemType const * problem, unsigned int width, Memory::MallocType mallocType) :
    width(width),
    problem(problem),
    cutset(width * problem->maxBranchingFactor, mallocType),
    currentStates(width, mallocType),
    nextStates(width, mallocType),
    auxiliaryData(width * problem->maxBranchingFactor, mallocType)
{
    // Cutset states
    LightArray<StateType> cutsetArray = cutset;
    unsigned int storageSize = StateType::sizeOfStorage(problem);
    std::byte* storages = StateType::getStorages(problem, cutsetArray.getCapacity(), mallocType);
    for (unsigned int cutsetStateIdx = 0; cutsetStateIdx < cutsetArray.getCapacity(); cutsetStateIdx += 1)
    {
        new (cutsetArray[cutsetStateIdx]) StateType(problem, storages);
        storages += storageSize;
    }

    // Current states
    LightArray<StateType> currentStatesArray = currentStates;
    storages = StateType::getStorages(problem, cutsetArray.getCapacity(), mallocType);
    for (unsigned int currentStateIdx = 0; currentStateIdx < currentStatesArray.getCapacity(); currentStateIdx += 1)
    {
        new (currentStatesArray[currentStateIdx])  StateType(problem, storages);
        storages += storageSize;
    }

    // Next states
    LightArray<StateType> nextStatesArray = nextStates;
    storages = StateType::getStorages(problem, cutsetArray.getCapacity(), mallocType);
    for (unsigned int nextStateIdx = 0; nextStateIdx < nextStatesArray.getCapacity(); nextStateIdx += 1)
    {
        new (nextStatesArray[nextStateIdx])  StateType(problem, storages);
        storages += storageSize;
    }
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::buildTopDown(DD::Type type, Neighbourhood const * neighbourhood)
{
    // Build
    bool cutsetSaved = false;
    unsigned int const variablesCount = problem->variables.getCapacity();
    for (unsigned int variableIdx = currentStates[0]->selectedValues.getSize(); variableIdx < variablesCount; variableIdx += 1)
    {
        resetAuxiliaryData();
        calcAuxiliaryData(variableIdx, neighbourhood);
        sortAuxiliaryData();
        shirkToFitAuxiliaryData();
        resizeNextStates(variableIdx, variablesCount);
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
        if (nextStates.isEmpty())
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
            saveCutset();
            cutsetSaved = true;
        }
        swapCurrentAndNextStates();
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
Vector<StateType> const * DD::MDD<ProblemType, StateType>::getCutset() const
{
    return &cutset;
}

template<typename ProblemType, typename StateType>
__host__ __device__
StateType* DD::MDD<ProblemType, StateType>::getBottom() const
{
    return currentStates[0];
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::setTop(StateType const * state)
{
    currentStates.clear();
    nextStates.clear();
    cutset.clear();

    currentStates.pushBack(state);
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::setInvalid()
{
#ifdef __CUDA_ARCH__
    if(threadIdx.x == 0)
#endif
    {
        getBottom()->setInvalid();
    }
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::calcAuxiliaryData(unsigned int variableIdx, Neighbourhood const * neighbourhood)
{
#ifdef __CUDA_ARCH__
    __syncthreads();
    unsigned int const currentStateIdx = threadIdx.x / problem->maxBranchingFactor;
    if (currentStateIdx < currentStates.getSize())
#else
    for(unsigned int currentStateIdx = 0; currentStateIdx < currentStates.getSize(); currentStateIdx += 1)
#endif
    {
        StateType const * const currentState = currentStates[currentStateIdx];
#ifdef __CUDA_ARCH__
        unsigned int const admissibleValueIdx = threadIdx.x % problem->maxBranchingFactor;
        if(admissibleValueIdx < currentState->admissibleValues.getSize())
#else
        for(unsigned int admissibleValueIdx = 0; admissibleValueIdx < currentState->admissibleValues.getSize(); admissibleValueIdx += 1)
#endif
        {
            OP::ValueType const value = *currentState->admissibleValues[admissibleValueIdx];
            bool boundsChk = problem->variables[variableIdx]->boundsCheck(value);
            bool constraintsChk = neighbourhood->constraintsCheck(variableIdx, value);
            if (boundsChk and constraintsChk)
            {
                unsigned int const valueIdx = value - problem->variables[variableIdx]->minValue;
                unsigned int const auxiliaryDataIdx = problem->maxBranchingFactor * currentStateIdx + valueIdx;
                auxiliaryData[auxiliaryDataIdx]->cost = calcCost(problem, currentState, value);
            }
        }
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::calcNextStates(unsigned int variableIdx)
{
#ifdef __CUDA_ARCH__
    __syncthreads();
    unsigned int const nextStateIdx = threadIdx.x;
    if(nextStateIdx < nextStates.getSize())
#else
    for(unsigned int nextStateIdx = 0; nextStateIdx < nextStates.getSize(); nextStateIdx += 1)
#endif
    {
        AuxiliaryData ad = *auxiliaryData[nextStateIdx];
        unsigned int const currentStateIdx = ad.index / problem->maxBranchingFactor ;
        unsigned int const valueIdx = ad.index % problem->maxBranchingFactor ;
        OP::ValueType const value = problem->variables[variableIdx]->minValue + valueIdx;
        makeState(problem, currentStates[currentStateIdx], value, auxiliaryData[nextStateIdx]->cost, nextStates[nextStateIdx]);
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::mergeNextStates(unsigned int variableIdx)
{
#ifdef __CUDA_ARCH__
    if(threadIdx.x == 0)
#endif
    {
        for (unsigned int auxiliaryDataIdx = nextStates.getSize(); auxiliaryDataIdx < auxiliaryData.getSize(); auxiliaryDataIdx += 1)
        {
            AuxiliaryData ad = *auxiliaryData[auxiliaryDataIdx];
            unsigned int const currentStateIdx = ad.index / problem->maxBranchingFactor;
            unsigned int const valueIdx = ad.index % problem->maxBranchingFactor;
            OP::ValueType const value = problem->variables[variableIdx]->minValue + valueIdx;
            mergeState(problem, currentStates[currentStateIdx], value, auxiliaryData[auxiliaryDataIdx]->cost, nextStates.back());
        }
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::resetAuxiliaryData()
{
    auxiliaryData.resize(currentStates.getSize() * problem->maxBranchingFactor); //resizeAuxiliaryData();
#ifdef __CUDA_ARCH__
    __syncthreads();
    unsigned int const auxiliaryDataIdx = threadIdx.x;
    if(auxiliaryDataIdx < auxiliaryData.getSize())
#else
    for(unsigned int auxiliaryDataIdx = 0; auxiliaryDataIdx < auxiliaryData.getSize(); auxiliaryDataIdx += 1)
#endif
    {
        auxiliaryData[auxiliaryDataIdx]->cost = DP::MaxCost;
        auxiliaryData[auxiliaryDataIdx]->index = auxiliaryDataIdx;
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::resizeAuxiliaryData()
{
#ifdef __CUDA_ARCH__
    if(threadIdx.x == 0)
#endif
    {
        auxiliaryData.resize(currentStates.getSize() * problem->maxBranchingFactor);
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::resizeCutset()
{
#ifdef __CUDA_ARCH__
    if(threadIdx.x == 0)
#endif
    {
        cutset.resize(nextStates.getSize());
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::resizeNextStates(unsigned int variableIdx, unsigned int variablesCount)
{
#ifdef __CUDA_ARCH__
    if(threadIdx.x == 0)
#endif
    {
        unsigned int nextStatesCount = width;
        if (variableIdx == variablesCount - 1)
        {
            nextStatesCount = 1;
        }
        nextStates.resize(min(nextStatesCount, auxiliaryData.getSize()));
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::saveCutset()
{
    cutset.resize(nextStates.getSize()); //resizeCutset();
#ifdef __CUDA_ARCH__
    __syncthreads();
    unsigned int const cutsetStateIdx = threadIdx.x;
    if(cutsetStateIdx < cutset.getSize())
#else
    for(unsigned int cutsetStateIdx = 0; cutsetStateIdx < cutset.getSize(); cutsetStateIdx += 1)
#endif
    {
        *cutset[cutsetStateIdx] = *nextStates[cutsetStateIdx];
    }
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::shirkToFitAuxiliaryData()
{
#ifdef __CUDA_ARCH__
    if(threadIdx.x == 0)
#endif
    {
        AuxiliaryData const invalidAuxiliaryData(DP::MaxCost, 0);
        AuxiliaryData const* const auxiliaryDataEnd = thrust::lower_bound(thrust::seq, auxiliaryData.begin(), auxiliaryData.end(), invalidAuxiliaryData);
        if (auxiliaryDataEnd != auxiliaryData.end())
        {
            unsigned int size = auxiliaryData.indexOf(auxiliaryDataEnd);
            auxiliaryData.resize(size);
        }
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::sortAuxiliaryData()
{
#ifdef __CUDA_ARCH__
    Algorithms::oddEvenSort(auxiliaryData.front(), auxiliaryData.getSize());
#else
    thrust::sort(thrust::seq, auxiliaryData.begin(), auxiliaryData.end());
#endif
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::swapCurrentAndNextStates()
{
#ifdef __CUDA_ARCH__
    if(threadIdx.x == 0)
#endif
    {
        LightVector<StateType>::swap(currentStates, nextStates);
    }
}