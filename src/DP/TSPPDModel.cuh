#pragma once

#include "../DD/AuxiliaryData.cuh"
#include "../OP/TSPPDProblem.cuh"
#include "TSPPDState.cuh"

namespace DP
{
    __host__ __device__ inline DP::CostType calcCost(OP::TSPPDProblem const * problem, TSPPDState const * currentState, OP::ValueType const value);
    __host__ __device__ inline void ifPickupAddDelivery(OP::ValueType value, TSPPDState* state);
    void makeRoot(OP::TSPPDProblem const * problem, TSPPDState* root);
    __host__ __device__ inline void makeState(OP::TSPPDProblem const * problem, TSPPDState const * currentState, OP::ValueType value, DP::CostType cost, TSPPDState* nextState);
    __host__ __device__ inline void mergeState(OP::TSPPDProblem const * problem, TSPPDState const * currentState, OP::ValueType value, TSPPDState* nextState);
}

__host__ __device__
DP::CostType DP::calcCost(OP::TSPPDProblem const * problem, TSPPDState const * currentState, OP::ValueType const value)
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


__host__ __device__
void DP::ifPickupAddDelivery(OP::ValueType value, TSPPDState* state)
{
    if (value % 2 == 0)
    {
        state->admissibleValuesMap.insert(value + 1);
    }
}


void DP::makeRoot(OP::TSPPDProblem const* problem, TSPPDState* root)
{
    root->cost = 0;
    for (u32 value = 0; value <= problem->maxValue; value += 2)
    {
        root->admissibleValuesMap.insert(value);
    }
}

__host__ __device__
void DP::makeState(OP::TSPPDProblem const * problem, TSPPDState const * currentState, OP::ValueType value, DP::CostType cost, TSPPDState* nextState)
{
    *nextState = *currentState;
    nextState->cost = cost;
    nextState->admissibleValuesMap.erase(value);
    nextState->selectValue(value);
    ifPickupAddDelivery(value, nextState);
}

__host__ __device__
void DP::mergeState(OP::TSPPDProblem const * problem, TSPPDState const * currentState, OP::ValueType value, TSPPDState* nextState)
{
    nextState->admissibleValuesMap.merge(currentState->admissibleValuesMap);
    ifPickupAddDelivery(value, nextState);
}