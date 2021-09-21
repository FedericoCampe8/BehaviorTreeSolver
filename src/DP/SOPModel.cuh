#pragma once

#include "../DD/StateMetadata.cuh"
#include "../OP/SOProblem.cuh"
#include "SOPState.cuh"

namespace DP
{
    __host__ __device__ inline DP::CostType calcCost(OP::SOProblem const * problem, SOPState const * currentState, OP::ValueType const value);
    void makeRoot(OP::SOProblem const * problem, SOPState* root);
    __host__ __device__ inline void makeState(OP::SOProblem const * problem, SOPState const * currentState, OP::ValueType value, DP::CostType cost, SOPState* nextState);
}

__host__ __device__
DP::CostType DP::calcCost(OP::SOProblem const * problem, SOPState const * currentState, OP::ValueType const value)
{
    if(not currentState->selectedValues.isEmpty())
    {
        OP::ValueType const from = *currentState->selectedValues.back();
        OP::ValueType const to = value;
        return currentState->cost + problem->getDistance(from, to);
    }
    else
    {
        return 0;
    }
}

void DP::makeRoot(OP::SOProblem const* problem, SOPState* root)
{
    //Initialize cost
    root->cost = 0;

    // Initialize precedences
    thrust::fill(thrust::seq, root->precedencesCount.begin(), root->precedencesCount.end(), 0);
    for (OP::ValueType from = 0; from <= problem->maxValue; from += 1)
    {
        for (OP::ValueType to = 0; to <= problem->maxValue; to += 1)
        {
            DP::CostType const distance = problem->getDistance(from, to);
            if (distance < 0)
            {
                *root->precedencesCount[from] += 1;
            }
        }
    }

    //Initialize admissible values
    root->admissibleValuesMap.insert(0);
}

__host__ __device__
void DP::makeState(OP::SOProblem const * problem, SOPState const * currentState, OP::ValueType value, DP::CostType cost, SOPState* nextState)
{
    // Generic
    *nextState = *currentState;
    nextState->cost = cost;
    nextState->admissibleValuesMap.erase(value);
    nextState->selectValue(value);

    // Update admissible values
    OP::ValueType to = value;
    for (OP::ValueType from = 0; from <= problem->maxValue; from += 1)
    {
        if (*nextState->precedencesCount[from] > 0)
        {
            if (problem->getDistance(from, to) < 0)
            {
                *nextState->precedencesCount[from] -= 1;
                if (*nextState->precedencesCount[from] == 0)
                {
                    nextState->admissibleValuesMap.insert(from);
                }
            }
        }
    }
}
