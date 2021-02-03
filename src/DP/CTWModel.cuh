#pragma once

#include <thrust/find.h>
#include <thrust/remove.h>
#include "../DD/AuxiliaryData.cuh"
#include "../OP/CTWProblem.cuh"
#include "CTWState.cuh"


namespace DP
{
    __host__ __device__ inline void calcAdmissibleValues(CTWState* state);
    __host__ __device__ inline DP::CostType calcCost(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType const value);
    __host__ __device__ inline OP::ValueType calcOtherEnd(OP::CTWProblem const * problem, OP::ValueType const value);
    __host__ __device__ inline bool closeInterruptedPair(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType const value);
    void makeRoot(OP::CTWProblem const * problem, CTWState* root);
    __host__ __device__ inline bool interruptPair(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType const value);
    __host__ __device__ inline void makeState(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType value, DP::CostType cost, CTWState* nextState);
    __host__ __device__ inline void mergeState(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType value, CTWState* nextState);
    __host__ __device__ inline void updateBlockingByAtomics(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
    __host__ __device__ inline void updateBlockingByDisjunctive1(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
    __host__ __device__ inline void updateBlockingByDisjunctive2(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
    __host__ __device__ inline u8 updatedS(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
    __host__ __device__ inline u8 updatedM(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
    __host__ __device__ inline u8 updatedL(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
    __host__ __device__ inline u8 updatedN(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
}

__host__ __device__
void DP::calcAdmissibleValues(DP::CTWState* state)
{
    state->admissibleValuesMap.clear();
    for (OP::ValueType value = 0; value < state->blockingConstraintsCount.getCapacity(); value += 1)
    {
        if((not state->selectedValuesMap.contains(value)) and *state->blockingConstraintsCount[value] <= 0)
        {
            state->admissibleValuesMap.insert(value);
        }
    }
}

__host__ __device__
DP::CostType DP::calcCost(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType const value)
{
    u32 s = 0;
    u32 m = 0;
    u32 l = 0;
    u32 n = 0;
    u32 const k = problem->k;

    if(not currentState->selectedValues.isEmpty())
    {
        s = updatedS(problem,currentState,value);
        m = updatedM(problem,currentState,value);
        l = updatedL(problem,currentState,value);
        n = updatedN(problem,currentState,value);
    }

    u32 const cost = (k * k * k * s) + (k * k * m) + (k * l) + n;
    return cost;
}

void DP::makeRoot(OP::CTWProblem const * problem, CTWState* root)
{
    thrust::fill(thrust::seq, root->blockingConstraintsCount.begin(), root->blockingConstraintsCount.end(), 0);
    for(Pair<OP::ValueType> const * atomicConstraint = problem->atomicConstraints.begin(); atomicConstraint != problem->atomicConstraints.end(); atomicConstraint += 1)
    {
        *root->blockingConstraintsCount[atomicConstraint->second] += 1;
    }
    calcAdmissibleValues(root);
}

__host__ __device__
void DP::makeState(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType value, DP::CostType cost, CTWState* nextState)
{
    *nextState = *currentState;
    nextState->cost = cost;

    nextState->s = updatedS(problem,nextState,value);
    nextState->m = updatedM(problem,nextState,value);
    nextState->l = updatedL(problem,nextState,value);
    nextState->n = updatedN(problem,nextState,value);

    if(interruptPair(problem, nextState, value))
    {
        Pair<OP::ValueType> const openPair(*nextState->selectedValues.back(), static_cast<OP::ValueType>(nextState->selectedValues.getSize() - 1));
        nextState->openPairs.pushBack(&openPair);
    }
    if(closeInterruptedPair(problem, nextState, value))
    {
        OP::ValueType const otherEnd = calcOtherEnd(problem,value);
        Pair<OP::ValueType> const * const openPairsEnd = thrust::remove_if(thrust::seq, nextState->openPairs.begin(), nextState->openPairs.end(), [=] __host__ __device__ (Pair<OP::ValueType> const & openPair) -> bool
        {
            return openPair.first == otherEnd;
        });
        nextState->openPairs.resize(openPairsEnd);
    }

    updateBlockingByAtomics(problem, nextState, value);
    updateBlockingByDisjunctive1(problem, nextState, value);
    updateBlockingByDisjunctive2(problem, nextState, value);
    nextState->selectValue(value);

    calcAdmissibleValues(nextState);
}

__host__ __device__
void DP::mergeState(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType value, CTWState* nextState)
{
    for (OP::ValueType value = 0; value < nextState->blockingConstraintsCount.getCapacity(); value += 1)
    {
        i32 const blockingConstraintsCount0 = *currentState->blockingConstraintsCount[value];
        i32 const blockingConstraintsCount1 = *nextState->blockingConstraintsCount[value];
        *nextState->blockingConstraintsCount[value] = static_cast<i8>(min(blockingConstraintsCount0,blockingConstraintsCount1));
    }

    updateBlockingByAtomics(problem, nextState, value);
    updateBlockingByDisjunctive1(problem, nextState, value);
    updateBlockingByDisjunctive2(problem, nextState, value);

    calcAdmissibleValues(nextState);
}


__host__ __device__
OP::ValueType DP::calcOtherEnd(OP::CTWProblem const * problem, OP::ValueType const value)
{
    if(value != 0)
    {
        return value <= problem->b ? value + problem->b : value - problem->b;
    }
    else
    {
        return 0;
    }
}

__host__ __device__
bool DP::closeInterruptedPair(OP::CTWProblem const * problem, DP::CTWState const * state, OP::ValueType const value)
{
    if(not state->selectedValues.isEmpty())
    {
        OP::ValueType const otherEnd = calcOtherEnd(problem, value);
        return *state->selectedValues.back() != otherEnd and state->selectedValuesMap.contains(otherEnd);
    }
    else
    {
        return false;
    }
}

__host__ __device__
bool DP::interruptPair(OP::CTWProblem const* problem, DP::CTWState const* state, OP::ValueType const value)
{
    if(not state->selectedValues.isEmpty())
    {
        OP::ValueType const lastSelectedValue = *state->selectedValues.back();
        OP::ValueType const lastSelectedValueOtherEnd = calcOtherEnd(problem, lastSelectedValue);
        OP::ValueType const otherEnd = calcOtherEnd(problem, value);
        return lastSelectedValue != otherEnd and (not state->selectedValuesMap.contains(lastSelectedValueOtherEnd));
    }
    else
    {
        return false;
    }
}

__host__ __device__
void DP::updateBlockingByAtomics(OP::CTWProblem const * problem, DP::CTWState const * state, OP::ValueType value)
{
    LightVector<u16> const * const atomicConstraintsMap = problem->atomicConstraintsMap[value];
    for(u16 const * atomicConstraintIdx = atomicConstraintsMap->begin(); atomicConstraintIdx != atomicConstraintsMap->end(); atomicConstraintIdx += 1)
    {
        Pair<OP::ValueType> const * const atomicConstraint = problem->atomicConstraints.at(*atomicConstraintIdx);
        OP::ValueType const & i = atomicConstraint->first;
        OP::ValueType const & j = atomicConstraint->second;
        if (value == i)
        {
            *state->blockingConstraintsCount[j] -= 1;
        }
    }
}

__host__ __device__
void DP::updateBlockingByDisjunctive1(OP::CTWProblem const* problem, DP::CTWState const* state, OP::ValueType value)
{
    LightVector<u16> const * const disjunctiveConstraints1Map = problem->disjunctiveConstraints1Map[value];
    for(u16 const * disjunctiveConstraint1Idx = disjunctiveConstraints1Map->begin(); disjunctiveConstraint1Idx != disjunctiveConstraints1Map->end(); disjunctiveConstraint1Idx += 1)
    {
        Triple<OP::ValueType> const * const disjunctiveConstraint1 = problem->disjunctiveConstraints1.at(*disjunctiveConstraint1Idx);
        OP::ValueType const & l = disjunctiveConstraint1->first;
        OP::ValueType const & i = disjunctiveConstraint1->second;
        OP::ValueType const & j = disjunctiveConstraint1->third;
        bool isPresentL = state->selectedValuesMap.contains(l);
        bool isPresentI = state->selectedValuesMap.contains(i);
        bool isPresentJ = state->selectedValuesMap.contains(j);
        if (value == i and (not isPresentL) and (not isPresentJ))
        {
            *state->blockingConstraintsCount[j] += 1;
        }
        else if(value == j and (not isPresentL) and (not isPresentI))
        {
            *state->blockingConstraintsCount[i] += 1;
        }
        else if (value == l)
        {
            if (isPresentI and (not isPresentJ))
            {
                *state->blockingConstraintsCount[j] -= 1;
            }
            else if ((not isPresentI) and isPresentJ)
            {
                *state->blockingConstraintsCount[i] -= 1;
            }
        }
    }
}

__host__ __device__
void DP::updateBlockingByDisjunctive2(OP::CTWProblem const* problem, DP::CTWState const* state, OP::ValueType value)
{
    LightVector<u16> const * const disjunctiveConstraints2Map = problem->disjunctiveConstraints2Map[value];
    for(u16 const * disjunctiveConstraint2Idx = disjunctiveConstraints2Map->begin(); disjunctiveConstraint2Idx != disjunctiveConstraints2Map->end(); disjunctiveConstraint2Idx += 1)
    {
        Triple<OP::ValueType> const * const disjunctiveConstraint2 = problem->disjunctiveConstraints2.at(*disjunctiveConstraint2Idx);
        OP::ValueType const & l = disjunctiveConstraint2->first;
        OP::ValueType const & i = disjunctiveConstraint2->second;
        OP::ValueType const & j = disjunctiveConstraint2->third;
        bool isPresentL = state->selectedValuesMap.contains(l);
        bool isPresentI = state->selectedValuesMap.contains(i);
        bool isPresentJ = state->selectedValuesMap.contains(j);
        if (value == i and  (not isPresentL))
        {
            *state->blockingConstraintsCount[l] += isPresentJ ? -1 : 1;
        }
        else if (value == j and (not isPresentL))
        {
            *state->blockingConstraintsCount[l] += isPresentI ? -1 : 1;
        }
    }
}

__host__ __device__
u8 DP::updatedS(OP::CTWProblem const* problem, DP::CTWState const* state, OP::ValueType value)
{
    u8 s = state->s;
    if(interruptPair(problem, state, value))
    {
        s += 1;
    }
    return s;
}

__host__ __device__
u8 DP::updatedM(OP::CTWProblem const * problem, DP::CTWState const * state, OP::ValueType value)
{
    u8 m = static_cast<u8>(max(state->m, state->openPairs.getSize()));
    return m;
}

__host__ __device__
u8 DP::updatedL(OP::CTWProblem const * problem, DP::CTWState const * state, OP::ValueType value)
{
    if (not state->openPairs.isEmpty())
    {
        u32 oldestOpenPairAge = state->selectedValues.getSize() - 1 - state->openPairs.front()->second;
        oldestOpenPairAge += calcOtherEnd(problem, value) == state->openPairs.front()->first ? 0 : 1; // closing oldest pair
        return static_cast<u8>(max(oldestOpenPairAge, state->l));
    }
    else
    {
        return 0;
    }
}

__host__ __device__
u8 DP::updatedN(OP::CTWProblem const* problem, DP::CTWState const* state, OP::ValueType value)
{
    u8 n = state->n;
    LightVector<u16> const * const softAtomicConstraintsMap = problem->softAtomicConstraintsMap[value];
    for(u16 const * softAtomicConstraintIdx = softAtomicConstraintsMap->begin(); softAtomicConstraintIdx != softAtomicConstraintsMap->end(); softAtomicConstraintIdx += 1)
    {
        Pair<OP::ValueType> const * const softAtomicConstraint = problem->softAtomicConstraints.at(*softAtomicConstraintIdx);
        OP::ValueType const & i = softAtomicConstraint->first;
        OP::ValueType const & j = softAtomicConstraint->second;
        bool isPresentI = state->selectedValuesMap.contains(i);
        bool isPresentJ = state->selectedValuesMap.contains(j);
        if ((not isPresentI) and isPresentJ)
        {
            n += 1;
        }
    }
    return n;
}

