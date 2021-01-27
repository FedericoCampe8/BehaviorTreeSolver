#pragma once

#include "../DD/AuxiliaryData.cuh"
#include "../OP/CTWProblem.cuh"
#include "CTWState.cuh"

namespace DP
{
    __host__ __device__ inline DP::CostType calcCost(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType const value);
    __host__ __device__ inline bool isPair(OP::CTWProblem const * problem, OP::ValueType const value0, OP::ValueType const value1);
    //__host__ __device__ inline void ifPickupAddDelivery(OP::VRProblem const * problem, OP::ValueType value, VRPState* state);
    //void makeRoot(OP::VRProblem const * problem, VRPState* root);    void makeRoot(OP::VRProblem const * problem, VRPState* root);
    //__host__ __device__ inline void makeState(OP::VRProblem const * problem, VRPState const * currentState, OP::ValueType value, DP::CostType cost, VRPState* nextState);
    //__host__ __device__ inline void mergeState(OP::VRProblem const * problem, VRPState const * currentState, OP::ValueType value, VRPState* nextState);
}

__host__ __device__
DP::CostType DP::calcCost(OP::CTWProblem const * problem, CTWState const * currentState, OP::ValueType const value)
{
    unsigned int s = 0;
    unsigned int m = 0;
    unsigned int l = 0;
    unsigned int n = 0;

    for (unsigned int pairIdx = 0; pairIdx < problem->b; pairIdx += 2)
    {
        int const selectedValuesSize = static_cast<int>(currentState->selectedValues.getSize());
        int const first = pairIdx == value ? selectedValuesSize : *currentState->pairs[pairIdx];
        int const second = pairIdx + 1 == value ? selectedValuesSize : *currentState->pairs[pairIdx + 1];

        s += abs(first - second) > 1 ? 1 : 0;

    }
    unsigned int const k = problem->k;
    return (k * k * k * s) + (k * k * m) + (k * l) + n;
}

__host__ __device__
bool DP::isPair(OP::CTWProblem const* problem, OP::ValueType const value0, OP::ValueType const value1)
{
    return abs(static_cast<int>(value0) - static_cast<int>(value1)) == problem->b;
}

    /*
    __host__ __device__
    void DP::ifPickupAddDelivery(OP::VRProblem const* problem, OP::ValueType value, VRPState* state)
    {
        OP::ValueType const * const pickup = thrust::find(thrust::seq, problem->pickups.begin(), problem->pickups.end(), value);
        if (pickup < problem->pickups.end())
        {
            unsigned int const deliveryIdx = problem->pickups.indexOf(pickup);
            OP::ValueType const * const delivery = problem->deliveries[deliveryIdx];
            if (not state->isAdmissible(*delivery))
            {
                state->admissibleValues.pushBack(delivery);
            }
        }
    }

    void DP::makeRoot(OP::VRProblem const* problem, VRPState* root)
    {
        root->cost = 0;
        root->selectedValues.pushBack(&problem->start);
        root->admissibleValues.pushBack(&problem->end);
        for (OP::ValueType const * pickup = problem->pickups.begin(); pickup != problem->pickups.end(); pickup += 1)
        {
            root->admissibleValues.pushBack(pickup);
        }
    }

    __host__ __device__
    void DP::makeState(OP::VRProblem const * problem, VRPState const * currentState, OP::ValueType value, DP::CostType cost, VRPState* nextState)
    {
        *nextState = *currentState;
        nextState->cost = cost;
        nextState->removeFromAdmissibles(value);
        nextState->selectedValues.pushBack(&value);
        ifPickupAddDelivery(problem, value, nextState);
    }

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