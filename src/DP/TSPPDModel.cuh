#pragma once

#include "../DD/StateMetadata.cuh"
#include "../OP/TSPPDProblem.cuh"
#include "TSPPDState.cuh"

namespace DP
{
    __host__ __device__ inline DP::CostType calcCost(OP::TSPPDProblem const * problem, TSPPDState const * currentState, OP::ValueType const value);
    __host__ __device__ inline void updateAdmissible(OP::ValueType value, TSPPDState* state);
    void makeRoot(OP::TSPPDProblem const * problem, TSPPDState* root);
    __host__ __device__ inline void makeState(OP::TSPPDProblem const * problem, TSPPDState const * currentState, OP::ValueType value, DP::CostType cost, TSPPDState* nextState);
}

__host__ __device__
DP::CostType DP::calcCost(OP::TSPPDProblem const * problem, TSPPDState const * currentState, OP::ValueType const value)
{
    if(currentState->selectedValues.getSize() > 1)
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


__host__ __device__
void DP::updateAdmissible(OP::ValueType value, TSPPDState* state)
{
    if(state->selectedValues.getSize() == 1)
    {
        for (u32 value = 2; value <= state->admissibleValuesMap.getMaxValue(); value += 2)
        {
            state->admissibleValuesMap.insert(value);
        }
    }
    else if(state->selectedValues.getSize() == state->selectedValues.getCapacity() - 1)
    {
        state->admissibleValuesMap.insert(1);
    }
    else if (value % 2 == 0)
    {
        state->admissibleValuesMap.insert(value + 1);
    }
}


void DP::makeRoot(OP::TSPPDProblem const* problem, TSPPDState* root)
{
    root->cost = 0;
    root->admissibleValuesMap.insert(0);
}

__host__ __device__
void DP::makeState(OP::TSPPDProblem const * problem, TSPPDState const * currentState, OP::ValueType value, DP::CostType cost, TSPPDState* nextState)
{
    *nextState = *currentState;
    nextState->cost = cost;
    nextState->admissibleValuesMap.erase(value);
    nextState->selectValue(value);
    updateAdmissible(value, nextState);
}