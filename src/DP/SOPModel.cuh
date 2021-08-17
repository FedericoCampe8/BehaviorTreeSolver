#pragma once

#include "../DD/StateMetadata.cuh"
#include "../OP/SOPProblem.cuh"
#include "SOPState.cuh"

namespace DP
{
    __host__ __device__ inline DP::CostType calcCost(OP::SOPProblem const * problem, SOPState const * currentState, OP::ValueType const value);
    __host__ __device__ inline void calcAdmissibleValues(OP::SOPProblem const * problem, SOPState* state);
    __host__ __device__ inline void updatePrecedencesCount(OP::SOPProblem const * problem, SOPState const * state, OP::ValueType value);
    void makeRoot(OP::SOPProblem const * problem, SOPState* root);
    __host__ __device__ inline void makeState(OP::SOPProblem const * problem, SOPState const * currentState, OP::ValueType value, DP::CostType cost, SOPState* nextState);
    __host__ __device__ inline void mergeState(OP::SOPProblem const * problem, SOPState const * currentState, OP::ValueType value, SOPState* nextState);
}

__host__ __device__
DP::CostType DP::calcCost(OP::SOPProblem const * problem, SOPState const * currentState, OP::ValueType const value)
{
    if(not currentState->selectedValues.isEmpty())
    {
        OP::ValueType const from = *currentState->selectedValues.back();
        OP::ValueType const to = value;
        return currentState->cost + problem->getDistance(to,from);
    }
    else
    {
        return 0;
    }
}


__host__ __device__
void DP::calcAdmissibleValues(OP::SOPProblem const * problem, SOPState* state)
{
    state->admissibleValuesMap.clear();

    for (u32 value = 0; value <= problem->maxValue; value += 1)
    {
        if(not state->selectedValuesMap.contains(value))
        {
            if (*state->precedencesCount[value] == 0)
            {
                state->admissibleValuesMap.insert(value);
            }
        }
    }
}

__host__ __device__
void DP::updatePrecedencesCount(OP::SOPProblem const* problem, DP::SOPState const* state, OP::ValueType value)
{
    OP::ValueType from = value;
    for (u32 to = 0; to <= problem->maxValue; to += 1)
    {
        DP::CostType const distance = problem->getDistance(from,to);
        if (distance < 0)
        {
            *state->precedencesCount[to] -= 1;
        }
    }
}


void DP::makeRoot(OP::SOPProblem const* problem, SOPState* root)
{
    root->cost = 0;
    thrust::fill(thrust::seq, root->precedencesCount.begin(), root->precedencesCount.end(), 0);
    for (u32 from = 0; from <= problem->maxValue; from += 1)
    {
        for (u32 to = 0; to <= problem->maxValue; to += 1)
        {
            DP::CostType const distance = problem->getDistance(from, to);
            if (distance < 0)
            {
                *root->precedencesCount[to] += 1;
            }
        }
    }
    calcAdmissibleValues(problem, root);
}

__host__ __device__
void DP::makeState(OP::SOPProblem const * problem, SOPState const * currentState, OP::ValueType value, DP::CostType cost, SOPState* nextState)
{
    *nextState = *currentState;
    nextState->cost = cost;
    nextState->selectValue(value);
    updatePrecedencesCount(problem, nextState, value);
    calcAdmissibleValues(problem, nextState);
}

__host__ __device__
void DP::mergeState(OP::SOPProblem const * problem, SOPState const * currentState, OP::ValueType value, SOPState* nextState)
{
    nextState->admissibleValuesMap.merge(currentState->admissibleValuesMap);
    //updateAdmissibles(problem, value, nextState);
}