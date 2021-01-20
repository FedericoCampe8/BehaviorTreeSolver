#pragma once

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/swap.h>
#include <Containers/Vector.cuh>
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
        private:
        unsigned int const width;
        ProblemType const * const problem;
        Vector<StateType> cutset;
        Vector<StateType> currentStates;
        Vector<StateType> nextStates;
        Vector<AuxiliaryData> auxiliaryData;

        // Functions
        public:
        MDD(ProblemType const* problem, unsigned int width, Memory::MallocType mallocType);
        __host__ __device__ void buildTopDown(DD::Type type, Neighbourhood const * neighbourhood);
        __host__ __device__ Vector<StateType> const * getCutset() const;
        __host__ __device__ StateType* getBottom() const;
        __host__ __device__ void setTop(StateType const * state);
        private:
        __host__ __device__ void calcAuxiliaryData(unsigned int variableIdx, Neighbourhood const * neighbourhood);
        __host__ __device__ void calcNextStates(unsigned int variableIdx);
        __host__ __device__ void mergeNextStates(unsigned int variableIdx);
        __host__ __device__ void resetAuxiliaryData();
        __host__ __device__ void resizeNextStates(unsigned int variableIdx, unsigned int variablesCount);
        __host__ __device__ void saveCutset();
        __host__ __device__ void shirkToFitAuxiliaryData();
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
    for (unsigned int cutsetStateIdx = 0; cutsetStateIdx < cutsetArray.getCapacity(); cutsetStateIdx += 1)
    {
        new (cutsetArray[cutsetStateIdx]) StateType(problem, mallocType);
    }

    // Current states
    LightArray<StateType> currentStatesArray = currentStates;
    for (unsigned int currentStateIdx = 0; currentStateIdx < currentStatesArray.getCapacity(); currentStateIdx += 1)
    {
        new (currentStatesArray[currentStateIdx]) StateType(problem, mallocType);
    }

    // Next states
    LightArray<StateType> nextStatesArray = nextStates;
    for (unsigned int nextStateIdx = 0; nextStateIdx < nextStatesArray.getCapacity(); nextStateIdx += 1)
    {
        new (nextStatesArray[nextStateIdx]) StateType(problem, mallocType);
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
        // Calculate auxiliary data
        resetAuxiliaryData();
        calcAuxiliaryData(variableIdx, neighbourhood);
        thrust::sort(thrust::seq, auxiliaryData.begin(), auxiliaryData.end());
        shirkToFitAuxiliaryData();
        if (auxiliaryData.isEmpty())
        {
            getBottom()->reset();
            return;
        }

        // Calculate next states
        resizeNextStates(variableIdx, variablesCount);
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

        //Prepare for the next loop
        LightVector<StateType>::swap(currentStates, nextStates);
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
void DD::MDD<ProblemType, StateType>::calcAuxiliaryData(unsigned int variableIdx, Neighbourhood const * neighbourhood)
{
    for(unsigned int currentStateIdx = 0; currentStateIdx < currentStates.getSize(); currentStateIdx += 1)
    {
        StateType const * const currentState = currentStates[currentStateIdx];
        for(unsigned int admissibleValueIdx = 0; admissibleValueIdx < currentState->admissibleValues.getSize(); admissibleValueIdx += 1)
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

    for(unsigned int nextStateIdx = 0; nextStateIdx < nextStates.getSize(); nextStateIdx += 1)
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
    for(unsigned int auxiliaryDataIdx = nextStates.getSize(); auxiliaryDataIdx < auxiliaryData.getSize(); auxiliaryDataIdx += 1)
    {
        AuxiliaryData ad = *auxiliaryData[auxiliaryDataIdx];
        unsigned int const currentStateIdx = ad.index / problem->maxBranchingFactor ;
        unsigned int const valueIdx = ad.index % problem->maxBranchingFactor ;
        OP::ValueType const value = problem->variables[variableIdx]->minValue + valueIdx;
        mergeState(problem, currentStates[currentStateIdx], value, auxiliaryData[auxiliaryDataIdx]->cost, nextStates.back());
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::resetAuxiliaryData()
{
    auxiliaryData.resize(currentStates.getSize() * problem->maxBranchingFactor);
    for(unsigned int auxiliaryDataIdx = 0; auxiliaryDataIdx < auxiliaryData.getSize(); auxiliaryDataIdx += 1)
    {
        auxiliaryData[auxiliaryDataIdx]->cost = DP::MaxCost;
        auxiliaryData[auxiliaryDataIdx]->index = auxiliaryDataIdx;
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::resizeNextStates(unsigned int variableIdx, unsigned int variablesCount)
{
    if (variableIdx < variablesCount - 1)
    {
        nextStates.resize(min(width, auxiliaryData.getSize()));
    }
    else
    {
        nextStates.resize(1);
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::saveCutset()
{
    cutset.resize(nextStates.getSize());
    for(unsigned int cutsetStateIdx = 0; cutsetStateIdx < cutset.getSize(); cutsetStateIdx += 1)
    {
        *cutset[cutsetStateIdx] = *nextStates[cutsetStateIdx];
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DD::MDD<ProblemType, StateType>::shirkToFitAuxiliaryData()
{
    AuxiliaryData const invalidMetadata(DP::MaxCost, 0);
    AuxiliaryData const* const auxiliaryDataEnd = thrust::lower_bound(thrust::seq, auxiliaryData.begin(), auxiliaryData.end(), invalidMetadata);
    if (auxiliaryDataEnd != auxiliaryData.end())
    {
        unsigned int size = auxiliaryData.indexOf(auxiliaryDataEnd);
        auxiliaryData.resize(size);
    }
}