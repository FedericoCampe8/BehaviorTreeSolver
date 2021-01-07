#include <thrust/find.h>

#include "VRPModel.cuh"

__global__
void DP::makeRoot(OP::VRProblem const * problem, VRPState* root)
{
    root->cost = 0;
    root->selectedValues.pushBack(&problem->start);
    root->admissibleValues.pushBack(&problem->end);
    for (OP::ValueType const * pickup = problem->pickups.begin(); pickup != problem->pickups.end(); pickup += 1)
    {
        root->admissibleValues.pushBack(pickup);
    }
}

__global__
void DP::calcCosts(OP::VRProblem const * problem, unsigned int variableIdx, Vector<VRPState>* currentStates, Vector<CostType>* costs)
{
    unsigned int const currentStateIdx = blockIdx.x;
    VRPState* const currentState = currentStates->at(currentStateIdx);
    unsigned int const admissibleValueIdx = threadIdx.x;
    if (admissibleValueIdx < currentState->admissibleValues.getSize())
    {
        OP::ValueType const value = *currentState->admissibleValues[admissibleValueIdx];
        OP::Variable const * const variable = problem->variables[variableIdx];
        if (variable->minValue <= value and value <= variable->maxValue)
        {
            OP::ValueType const from = *currentState->selectedValues.back();
            OP::ValueType const to = value;
            unsigned int valueIdx = value - problem->variables[variableIdx]->minValue;
            DP::CostType* const cost = costs->at(problem->maxBranchingFactor * currentStateIdx + valueIdx);
            *cost = currentState->cost + problem->getDistance(from, to);
        }
    }
}

__global__
void DP::makeStates(OP::VRProblem const * problem, unsigned int variableIdx, Vector<VRPState> const * currentStates, Vector<VRPState> const * nextStates, Vector<uint32_t> const * indices, Vector<CostType> const * costs)
{
    unsigned int nextStateIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if(nextStateIdx < nextStates->getSize())
    {
        uint32_t const index = *indices->at(nextStateIdx);
        unsigned int const currentStateIdx = index / problem->maxBranchingFactor;
        OP::ValueType const valueIdx = index % problem->maxBranchingFactor;
        OP::ValueType const value = problem->variables[variableIdx]->minValue + valueIdx;
        VRPState* const nextState = nextStates->at(nextStateIdx);

        // Initialize next state
        *nextState = *currentStates->at(currentStateIdx);
        nextState->cost = *costs->at(nextStateIdx);

        // Remove value from admissible values
        nextState->removeFromAdmissibles(value);

        // Add value to selected values
        nextState->selectedValues.pushBack(&value);

        // If pickup add delivery
        OP::ValueType const * const pickup = thrust::find(thrust::seq, problem->pickups.begin(), problem->pickups.end(), value);
        if (pickup < problem->pickups.end())
        {
            unsigned int const deliveryIdx = problem->pickups.indexOf(pickup);
            OP::ValueType const delivery = *problem->deliveries[deliveryIdx];
            if(not nextState->isAdmissible(delivery))
            {
                nextState->admissibleValues.pushBack(&delivery);
            }
        }
    }
}