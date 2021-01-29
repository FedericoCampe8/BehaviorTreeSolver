#pragma once

#include <thrust/find.h>
#include <thrust/remove.h>
#include "../DD/AuxiliaryData.cuh"
#include "../OP/CTWProblem.cuh"
#include "CTWState.cuh"

namespace DP
{
    __host__ __device__ inline DP::CostType calcCost(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType const value);

    void makeRoot(OP::CTWProblem const * problem, CTWState* root);
    __host__ __device__ inline void makeState(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType value, DP::CostType cost, CTWState* nextState);
    //__host__ __device__ inline void mergeState(OP::VRProblem const * problem, VRPState const * currentState, OP::ValueType value, VRPState* nextState);
    __host__ __device__ inline OP::ValueType calcOtherEnd(OP::CTWProblem const * problem, OP::ValueType const value);
    __host__ __device__ void closePair(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType const value);
    __host__ __device__ bool pairClosed(OP::CTWProblem const * problem,CTWState const * currentState, OP::ValueType const value);
    __host__ __device__ bool pairInterrupted(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType const value);
    __host__ __device__ void updateByAtomics(OP::CTWProblem const * problem, CTWState const * nextState, OP::ValueType const value);
    __host__ __device__ void updateByDisjunctive1(OP::CTWProblem const * problem, CTWState const * nextState, OP::ValueType const value);
}

__host__ __device__
DP::CostType DP::calcCost(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType const value)
{
    unsigned int s = currentState->s;
    unsigned int m = currentState->m;
    unsigned int l = 0;
    unsigned int n = currentState->n;
    unsigned int const k = problem->k;

    if(not currentState->selectedValues.isEmpty())
    {
        if(pairInterrupted(problem, currentState, value))
        {
            s += 1;
            m = max(m, currentState->openPairsCount + 1);
        }

        l = currentState->selectedValues.getSize() - currentState->oldestOpenPairIdx;
        bool closeOldestOpenedPair = calcOtherEnd(problem, value) == *currentState->selectedValues[currentState->oldestOpenPairIdx];
        l -= closeOldestOpenedPair ? 1 : 0;

        LightVector<unsigned int> const * const softAtomicConstraintsMap = problem->softAtomicConstraintsMap[value];
        for(unsigned int softAtomicConstraintIdx = 0; softAtomicConstraintIdx < softAtomicConstraintsMap->getSize(); softAtomicConstraintIdx += 1)
        {
            Pair<OP::ValueType> const * const softAtomicConstraint = problem->softAtomicConstraints[softAtomicConstraintIdx];
            bool firstPresent = currentState->isSelected(softAtomicConstraint->first);
            bool secondPresent = currentState->isSelected(softAtomicConstraint->second);
            n += softAtomicConstraint->first == value and secondPresent ? 1 : 0;
            n += softAtomicConstraint->second == value and firstPresent ? 0 : 1;
        }
    }
    return (k * k * k * s) + (k * k * m) + (k * l) + n;
}

void DP::makeRoot(OP::CTWProblem const * problem, CTWState* root)
{
    thrust::fill(thrust::seq, root->blockingConstraintsCount.begin(), root->blockingConstraintsCount.end(), 0);
    for(unsigned int atomicConstraintIdx = 0; atomicConstraintIdx < problem->atomicConstraints.getSize(); atomicConstraintIdx += 1)
    {
        *root->blockingConstraintsCount[problem->atomicConstraints[atomicConstraintIdx]->second] += 1;
    }
    for(OP::ValueType value = 0; value < root->blockingConstraintsCount.getCapacity(); value += 1)
    {
        if(*root->blockingConstraintsCount[value] == 0)
        {
            root->admissibleValues.pushBack(&value);
        }
    }
}


__host__ __device__
void DP::makeState(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType value, DP::CostType cost, CTWState* nextState)
{
    *nextState = *currentState;
    nextState->cost = cost;
    nextState->selectedValues.pushBack(&value);

    // Cost related
    if(pairClosed(problem, currentState, value))
    {
        closePair(problem, nextState,value);
    }
    else
    {
        if(pairInterrupted(problem, currentState, value))
        {
            nextState->openPairsCount += 1;
            nextState->s += 1;
            nextState->m = max(nextState->m, currentState->openPairsCount);
        }
    }
    updateByAtomics(problem,nextState,value);



}


/*
__host__ __device__
void DP::mergeState(OP::VRProblem const * problem, VRPState const * currentState, OP::ValueType value, VRPState* nextState)
{
    for (OP::ValueType const* admissibleValue = currentState->admissibleValues.begin(); admissibleValue != currentState->admissibleValues.end(); admissibleValue += 1)
    {
        if (* admissibleValue != value and (not nextState->isAdmissible(* admissibleValue)))
        {
            nextState->admissibleValues.pushBack(admissibleValue);
        }
    }
    ifPickupAddDelivery(problem, value, nextState);
}
 */

__host__ __device__
OP::ValueType DP::calcOtherEnd(OP::CTWProblem const * problem, OP::ValueType const value)
{
    return value < problem->b ? value + problem->b : value - problem->b;
}

__host__ __device__
void DP::closePair(OP::CTWProblem const * problem, DP::CTWState const * currentState, OP::ValueType const value)
{
    OP::ValueType const otherEnd = calcOtherEnd(problem,value);
    OP::ValueType const * const openedPairsEnd = thrust::remove_if(thrust::seq, currentState->openedPairs.begin(), currentState->openedPairs.end(), [=] __host__ __device__ (Pair<OP::ValueType> const & pair) -> bool
    {
        pair.first == otherEnd;
    });
    currentState->openedPairs.resize(openedPairsEnd);
}


__host__ __device__
bool DP::pairClosed(OP::CTWProblem const * problem, DP::CTWState const * currentState, OP::ValueType const value)
{
    OP::ValueType const otherEnd = calcOtherEnd(problem, value);
    return thrust::find(thrust::seq, currentState->selectedValues.begin(), currentState->selectedValues.end(), otherEnd) != currentState->selectedValues.end();
}

__host__ __device__
bool DP::pairInterrupted(OP::CTWProblem const* problem, DP::CTWState const* currentState, OP::ValueType const value)
{
    OP::ValueType const otherEnd = calcOtherEnd(problem, value);
    return *currentState->selectedValues.back() == otherEnd;
}

__host__ __device__ void
DP::updateByAtomics(OP::CTWProblem const* problem, DP::CTWState const* nextState, OP::ValueType const value)
{
    LightVector<unsigned int> const * const atomicConstraintsMap = problem->atomicConstraintsMap[value];
    for(unsigned int atomicConstraintIdx = 0; atomicConstraintIdx < atomicConstraintsMap->getSize(); atomicConstraintIdx += 1)
    {
        Pair<OP::ValueType> const * const atomicConstraint = problem->atomicConstraints[atomicConstraintIdx];
        if(value == atomicConstraint->first)
        {
            nextState->blockingConstraintsCount[atomicConstraint->second] -= 1;
        }
    }
}
__host__ __device__
void DP::updateByDisjunctive1(OP::CTWProblem const* problem, DP::CTWState const* nextState, OP::ValueType const value)
{
    LightVector<unsigned int> const * const disjunctiveConstraints1Map = problem->disjunctiveConstraints1Map[value];
    for(unsigned int disjunctiveConstraintIdx = 0; disjunctiveConstraintIdx < disjunctiveConstraints1Map->getSize(); disjunctiveConstraintIdx += 1)
    {
        Triple<OP::ValueType> const * const disjunctiveConstraint = problem->disjunctiveConstraints1[disjunctiveConstraintIdx];
        bool lSelected = nextState->isSelected(disjunctiveConstraint->first);
        bool iSelected = nextState->isSelected(disjunctiveConstraint->second);
        bool jSelected = nextState->isSelected(disjunctiveConstraint->third);
        if(value == disjunctiveConstraint->first) // l
        {
            if(not iSelected)
            {
                nextState->blockingConstraintsCount[disjunctiveConstraint->second] -= 1;
            }

            if(not jSelected)
            {
                nextState->blockingConstraintsCount[disjunctiveConstraint->third] -= 1;
            }
        }

        if(value == disjunctiveConstraint->second) // i
        {
            if(not nextState->isSelected(disjunctiveConstraint->first))
            {
                currentState->blockingConstraintsCount[disjunctiveConstraint->first] -= 1;
            }
        }
    }
}
