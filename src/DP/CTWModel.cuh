#pragma once

#include <thrust/find.h>
#include <thrust/remove.h>
#include "../DD/AuxiliaryData.cuh"
#include "../OP/CTWProblem.cuh"
#include "CTWState.cuh"


namespace DP
{
    __host__ __device__ inline void calcAdmissibleValues(OP::CTWProblem const * problem, CTWState* state);
    __host__ __device__ inline DP::CostType calcCost(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType const value);
    __host__ __device__ inline OP::ValueType calcOtherEnd(OP::CTWProblem const * problem, OP::ValueType const value);
    __host__ __device__ inline bool closeInterruptedPair(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType const value);
    __host__ __device__ inline bool isClosingPair(OP::CTWProblem const * problem, CTWState const * state);
    void makeRoot(OP::CTWProblem const * problem, CTWState* root);
    __host__ __device__ inline bool interruptPair(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType const value);
    __host__ __device__ inline void makeState(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType value, DP::CostType cost, CTWState* nextState);
    __host__ __device__ inline void mergeState(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType value, CTWState* nextState);
    __host__ __device__ inline void updateInterruptedPairs(OP::CTWProblem const * problem, CTWState* state, OP::ValueType value);
    __host__ __device__ inline void updatePrecedencesCount(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
    __host__ __device__ inline bool checkDisjunctive1(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
    __host__ __device__ inline bool checkDisjunctive2(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
    __host__ __device__ inline bool checkPrecedence(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType i, OP::ValueType j);
    __host__ __device__ inline u8 updatedS(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
    __host__ __device__ inline u8 updatedM(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
    __host__ __device__ inline u8 updatedL(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
    __host__ __device__ inline u8 updatedN(OP::CTWProblem const * problem, CTWState const * state, OP::ValueType value);
}

__host__ __device__
void DP::calcAdmissibleValues(OP::CTWProblem const * problem, DP::CTWState* state)
{
    state->admissibleValuesMap.clear();
    for (OP::ValueType value = 0; value < state->precedencesCount.getCapacity(); value += 1)
    {
        if(not state->selectedValuesMap.contains(value))
        {
            if (*state->precedencesCount[value] == 0)
            {
                if (checkDisjunctive1(problem, state, value))
                {
                    if (checkDisjunctive2(problem, state, value))
                    {
                        state->admissibleValuesMap.insert(value);
                    }
                }
            }
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
        s = updatedS(problem, currentState, value);
        m = updatedM(problem,currentState,value);
        l = updatedL(problem,currentState,value);
        n = updatedN(problem,currentState,value);
    }

    u32 const cost = (k * k * k * s) + (k * k * m) + (k * l) + n;
    return cost;
}

void DP::makeRoot(OP::CTWProblem const* problem, DP::CTWState* root)
{
    thrust::fill(thrust::seq, root->precedencesCount.begin(), root->precedencesCount.end(), 0);
    for(Pair<OP::ValueType> const * atomicConstraint = problem->atomicConstraints.begin();  atomicConstraint != problem->atomicConstraints.end(); atomicConstraint += 1)
    {
        OP::ValueType const & i = atomicConstraint->first;
        *root->precedencesCount[i] += 1;
    }
    calcAdmissibleValues(problem, root);
}


__host__ __device__
void DP::makeState(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType value, DP::CostType cost, CTWState* nextState)
{
    *nextState = *currentState;
    nextState->cost = cost;
    nextState->selectValue(value);

    nextState->s = updatedS(problem, currentState, value);
    nextState->m = updatedM(problem, currentState, value);
    nextState->l = updatedL(problem, currentState, value);
    nextState->n = updatedN(problem, currentState, value);
    updateInterruptedPairs(problem, nextState, value);

    updatePrecedencesCount(problem, nextState, value);
    calcAdmissibleValues(problem, nextState);

    u32 tmp [] = {0,4,12,8,16,13,10,2,3,7,15,6,14,5,11,1,9};
    bool eq = true;
    u32 i = 0;
    for (OP::ValueType const* value = nextState->selectedValues.begin(); value != nextState->selectedValues.end(); value += 1)
    {
        eq = eq and (*value == tmp[i]);
        i +=1;
    }
    if(eq)
    {
        nextState->print();
    }
}

__host__ __device__
void DP::mergeState(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType value, CTWState* nextState)
{
    for (OP::ValueType value = 0; value < nextState->precedencesCount.getCapacity(); value += 1)
    {
        u32 const blockingConstraintsCount0 = *currentState->precedencesCount[value];
        u32 const blockingConstraintsCount1 = *nextState->precedencesCount[value];
        *nextState->precedencesCount[value] = static_cast<u8>(min(blockingConstraintsCount0,blockingConstraintsCount1));
    }

    updatePrecedencesCount(problem, nextState, value);
    calcAdmissibleValues(problem, nextState);
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
        // Selected values = i,...
        OP::ValueType const otherEnd = calcOtherEnd(problem, value);
        bool const isPresentOtherEnd = state->selectedValuesMap.contains(otherEnd);
        return isPresentOtherEnd and otherEnd != *state->selectedValues.back();
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
        // Selected values = i,...
        OP::ValueType const i = *state->selectedValues.back();
        OP::ValueType const j = calcOtherEnd(problem, i);
        bool const isPresentJ = state->selectedValuesMap.contains(j);
        if ((not isPresentJ) and value != j)
        {
           return true;
        }
    }
    return false;
}

__host__ __device__
void DP::updatePrecedencesCount(OP::CTWProblem const* problem, DP::CTWState const* state, OP::ValueType value)
{
    LightVector<u16> const * const atomicConstraintsMap = problem->atomicToCheck[value];
    for(u16 const * atomicConstraintIdx = atomicConstraintsMap->begin(); atomicConstraintIdx != atomicConstraintsMap->end(); atomicConstraintIdx += 1)
    {
        Pair<OP::ValueType> const * const atomicConstraint = problem->atomicConstraints.at(*atomicConstraintIdx);
        OP::ValueType const & i = atomicConstraint->first;
        OP::ValueType const & j = atomicConstraint->second;
        if (value == j)
        {
            *state->precedencesCount[i] -= 1;
        }
    }
}

__host__ __device__
bool DP::checkDisjunctive1(OP::CTWProblem const* problem, DP::CTWState const* state, OP::ValueType value)
{
    LightVector<u16> const * const disjunctive1ToCheck = problem->disjunctive1ToCheck[value];
    for(u16 const * disjunctiveConstraint1Idx = disjunctive1ToCheck->begin(); disjunctiveConstraint1Idx != disjunctive1ToCheck->end(); disjunctiveConstraint1Idx += 1)
    {
        Triple<OP::ValueType> const * const disjunctiveConstraint1 = problem->disjunctiveConstraints.at(*disjunctiveConstraint1Idx);
        OP::ValueType const & i = disjunctiveConstraint1->second;
        OP::ValueType const & j = disjunctiveConstraint1->third;
        bool const isPresentI = state->selectedValuesMap.contains(i);
        bool const isPresentJ = state->selectedValuesMap.contains(j);
        if ((not isPresentI) and (not isPresentJ))
        {
            return false;
        }
    }
    return true;
}

__host__ __device__
bool DP::checkDisjunctive2(OP::CTWProblem const* problem, DP::CTWState const* state, OP::ValueType value)
{
    LightVector<u16> const * const disjunctive2ToCheck = problem->disjunctive2ToCheck[value];
    for(u16 const * disjunctiveConstraint2Idx = disjunctive2ToCheck->begin(); disjunctiveConstraint2Idx != disjunctive2ToCheck->end(); disjunctiveConstraint2Idx += 1)
    {
        Triple<OP::ValueType> const * const disjunctiveConstraint2 = problem->disjunctiveConstraints.at(*disjunctiveConstraint2Idx);
        OP::ValueType const & i = disjunctiveConstraint2->second;
        OP::ValueType const & j = disjunctiveConstraint2->third;
        bool const isPresentI = state->selectedValuesMap.contains(i);
        bool const isPresentJ = state->selectedValuesMap.contains(j);
        if (isPresentJ and (not isPresentI))
        {
            return false;
        }
    }
    return true;
}

__host__ __device__
u8 DP::updatedS(OP::CTWProblem const* problem, DP::CTWState const * state, OP::ValueType value)
{
    u8 s = state->s;
    if (interruptPair(problem,state,value))
    {
        s += 1;
    }
    return s;
}

__host__ __device__
u8 DP::updatedM(OP::CTWProblem const * problem, DP::CTWState const * state, OP::ValueType value)
{
    u32 interruptedPairCount = state->interruptedPairs.getSize();
    if(closeInterruptedPair(problem,state,value)) // Closing interrupted pair
    {
        interruptedPairCount -= 1;
    }
    return static_cast<u8>(max(state->m, interruptedPairCount));

}

__host__ __device__
u8 DP::updatedL(OP::CTWProblem const * problem, DP::CTWState const * state, OP::ValueType value)
{
    if (not state->interruptedPairs.isEmpty())
    {
        u32 oldestInterruptedPairAge = state->selectedValues.getSize() - 1 - state->interruptedPairs.front()->second;
        if(calcOtherEnd(problem, value) != state->interruptedPairs.front()->first) // Not closing oldest interrupted pair
        {
            oldestInterruptedPairAge +=  1;
        }
        return static_cast<u8>(max(oldestInterruptedPairAge, state->l));
    }
    else
    {
        return state->l;
    }
}

__host__ __device__
u8 DP::updatedN(OP::CTWProblem const* problem, DP::CTWState const* state, OP::ValueType value)
{
    u8 n = state->n;
    LightVector<u16> const * const softAtomicConstraintsMap = problem->softAtomicToCheck[value];
    for(u16 const * softAtomicConstraintIdx = softAtomicConstraintsMap->begin(); softAtomicConstraintIdx != softAtomicConstraintsMap->end(); softAtomicConstraintIdx += 1)
    {
        Pair<OP::ValueType> const * const softAtomicConstraint = problem->softAtomicConstraints.at(*softAtomicConstraintIdx);
        OP::ValueType const & j = softAtomicConstraint->second;
        bool isPresentJ = state->selectedValuesMap.contains(j);
        if (not isPresentJ)
        {
            n += 1;
        }
    }
    return n;
}

__host__ __device__
void DP::updateInterruptedPairs(OP::CTWProblem const* problem, DP::CTWState* state, OP::ValueType value)
{
    if(interruptPair(problem, state, value))
    {
        Pair<OP::ValueType> const openPair(*state->selectedValues.back(), static_cast<OP::ValueType>(state->selectedValues.getSize() - 1));
        state->interruptedPairs.pushBack(&openPair);
    }
    if(closeInterruptedPair(problem, state, value))
    {
        OP::ValueType const otherEnd = calcOtherEnd(problem,value);
        Pair<OP::ValueType> const * const openPairsEnd = thrust::remove_if(thrust::seq, state->interruptedPairs.begin(), state->interruptedPairs.end(), [=] __host__ __device__ (Pair<OP::ValueType> const & openPair) -> bool
        {
            return openPair.first == otherEnd;
        });
        state->interruptedPairs.resize(openPairsEnd);
    }
}
