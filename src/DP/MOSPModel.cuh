#pragma once

#include "../DD/AuxiliaryData.cuh"
#include "../OP/MOSProblem.cuh"
#include "MOSPState.cuh"

namespace DP
{
    __host__ __device__ inline DP::CostType calcCost(OP::MOSProblem const * problem, MOSPState const * currentState, OP::ValueType const value);
    void makeRoot(OP::MOSProblem const * problem, MOSPState* root);
    __host__ __device__ inline void makeState(OP::MOSProblem const * problem, MOSPState const * currentState, OP::ValueType value, DP::CostType cost, MOSPState* nextState);
    __host__ __device__ inline void mergeState(OP::MOSProblem const * problem, MOSPState const * currentState, OP::ValueType value, MOSPState* nextState);
}

__host__ __device__
DP::CostType DP::calcCost(OP::MOSProblem const * problem, MOSPState const * currentState, OP::ValueType const value)
{
    u16 openStacksCount = 0;
    OP::ValueType const product = value;
    for (OP::ValueType client = 0; client < problem->clients; client += 1)
    {
        if(problem->getOrder(client, product) > 0)
        {
            if (*currentState->productsToDo[client] == 0)
            {
                    openStacksCount += 1;
            }
        }
    }
    DP::CostType cost = currentState->openStacksCount + openStacksCount - currentState->clientsToClose;
    return Algorithms::max(currentState->cost, cost);
}


void DP::makeRoot(OP::MOSProblem const* problem, MOSPState* root)
{
    root->cost = 0;
    for (OP::ValueType client = 0; client < problem->clients; client += 1)
    {
        *root->productsToDo[client] = 0;
    }
    for (OP::ValueType product = 0; product < problem->products; product += 1)
    {
        root->admissibleValuesMap.insert(product);
    }
}

__host__ __device__
void DP::makeState(OP::MOSProblem const * problem, MOSPState const * currentState, OP::ValueType value, DP::CostType cost, MOSPState* nextState)
{
    // Generic
    *nextState = *currentState;
    nextState->cost = cost;
    nextState->admissibleValuesMap.erase(value);
    nextState->selectValue(value);

    // Problem specific
    nextState->openStacksCount -= nextState->clientsToClose;
    nextState->clientsToClose = 0;

    OP::ValueType const product = value;
    for (OP::ValueType client = 0; client < problem->clients; client += 1)
    {
        if(problem->getOrder(client, product) > 0)
        {
            if (*nextState->productsToDo[client] == 0)
            {
                for (OP::ValueType p = 0; p < problem->products; p += 1)
                {
                    *nextState->productsToDo[client] += problem->getOrder(client, p) != 0 ? 1 : 0;
                }
                nextState->openStacksCount += 1;
            }

            *nextState->productsToDo[client] -= 1;

            if (*nextState->productsToDo[client] == 0)
            {
                nextState->clientsToClose += 1;
            }
        }
    }
    //nextState->print(true);
}

__host__ __device__
void DP::mergeState(OP::MOSProblem const * problem, MOSPState const * currentState, OP::ValueType value, MOSPState* nextState)
{
    nextState->admissibleValuesMap.merge(currentState->admissibleValuesMap);
    nextState->openStacksCount = Algorithms::min(nextState->openStacksCount, currentState->openStacksCount);
    nextState->maxOpenStacksCount = Algorithms::min(nextState->maxOpenStacksCount, currentState->maxOpenStacksCount);
    nextState->clientsToClose = Algorithms::min(nextState->clientsToClose, currentState->clientsToClose);
    for (OP::ValueType client = 0; client < problem->clients; client += 1)
    {
       *nextState->productsToDo[client] = Algorithms::min(*nextState->productsToDo[client], *currentState->productsToDo[client]);
    }
}