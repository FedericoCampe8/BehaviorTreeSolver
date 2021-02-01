#pragma once

#include "../DD/AuxiliaryData.cuh"
#include "../OP/VRProblem.cuh"
#include "VRPState.cuh"

namespace DP
{
    __host__ __device__ inline DP::CostType calcCost(OP::VRProblem const * problem, VRPState const * currentState, OP::ValueType const value);
    __host__ __device__ inline void ifPickupAddDelivery(OP::ValueType value, VRPState* state);
    void makeRoot(OP::VRProblem const * problem, VRPState* root);    void makeRoot(OP::VRProblem const * problem, VRPState* root);
    __host__ __device__ inline void makeState(OP::VRProblem const * problem, VRPState const * currentState, OP::ValueType value, DP::CostType cost, VRPState* nextState);
    __host__ __device__ inline void mergeState(OP::VRProblem const * problem, VRPState const * currentState, OP::ValueType value, VRPState* nextState);
}

__host__ __device__
DP::CostType DP::calcCost(OP::VRProblem const * problem, VRPState const * currentState, OP::ValueType const value)
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
void DP::ifPickupAddDelivery(OP::ValueType value, VRPState* state)
{
    if (value % 2 == 0)
    {
        state->admissibleValuesMap.insert(value + 1);
    }
}


void DP::makeRoot(OP::VRProblem const* problem, VRPState* root)
{
    root->cost = 0;
    for (OP::ValueType value = 0; value <= problem->maxValue; value += 2)
    {
        root->admissibleValuesMap.insert(value);
    }
}

__host__ __device__
void DP::makeState(OP::VRProblem const * problem, VRPState const * currentState, OP::ValueType value, DP::CostType cost, VRPState* nextState)
{
    *nextState = *currentState;
    nextState->cost = cost;
    nextState->admissibleValuesMap.erase(value);
    nextState->selectValue(value);
    ifPickupAddDelivery(value, nextState);
}

__host__ __device__
void DP::mergeState(OP::VRProblem const * problem, VRPState const * currentState, OP::ValueType value, VRPState* nextState)
{
    nextState->admissibleValuesMap.merge(currentState->admissibleValuesMap);
    ifPickupAddDelivery(value, nextState);
}