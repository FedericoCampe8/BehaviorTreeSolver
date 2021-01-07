#include <thrust/binary_search.h>
#include <thrust/equal.h>

#include "VRPModel.cuh"

DP::VRPModel::VRPModel(OP::VRProblem const * problem) :
    problem(problem)
{}


void DP::VRPModel::makeRoot(VRPState* root) const
{
    root->cost = 0;
    root->admissibleValues.pushBack(&problem->start);
    root->admissibleValues.pushBack(&problem->end);
    for (OP::ValueType const * pickup = problem->pickups.begin(); pickup != problem->pickups.end(); pickup += 1)
    {
        root->admissibleValues.pushBack(pickup);
    }
}

__host__ __device__
void DP::VRPModel::calcCosts(unsigned int variableIdx, VRPState const * state, LNS::Neighbourhood const * neighbourhood, CostType* costs) const
{
    OP::Variable const * const variable = problem->variables[variableIdx];
    for(OP::ValueType const * value = state->admissibleValues.begin(); value != state->admissibleValues.end(); value += 1)
    {
        if (variable->minValue <= *value and *value <= variable->maxValue)
        {
            LNS::ConstraintType const variableConstraint = *neighbourhood->constraints[variableIdx];
            bool const condition0 = variableConstraint == LNS::ConstraintType::None and (not *neighbourhood->constrainedValues[*value]);
            bool const condition1 = variableConstraint == LNS::ConstraintType::Eq and *neighbourhood->solution[variableIdx] == *value;
            bool const condition2 = variableConstraint == LNS::ConstraintType::Neq and *neighbourhood->solution[variableIdx] != *value;

            if (condition0 or condition1 or condition2)
            {
                DP::CostType edgeCost = not state->selectedValues.isEmpty() ? problem->getDistance(*state->selectedValues.back(), *value) : 0;
                unsigned int edgeIdx = *value - variable->minValue;
                costs[edgeIdx] = state->cost + edgeCost;
            }
        }
    }
}

__host__ __device__
void DP::VRPModel::makeState(VRPState const * parentState, OP::ValueType value, unsigned int childStateCost, VRPState* childState) const
{
    // Initialize child state
    *childState = *parentState;
    childState->cost = childStateCost;

    // Remove value from admissible values
    assert(parentState->isAdmissible(value));
    childState->removeFromAdmissibles(value);

    // Add value to selected values
    childState->selectedValues.pushBack(&value);

    ifPickupAddDelivery(value, childState);
}

__host__ __device__
void DP::VRPModel::mergeState(VRPState const * parentState, OP::ValueType value, VRPState* childState) const
{
    // Merge admissible values
    for (OP::ValueType const * admissibleValue = parentState->admissibleValues.begin(); admissibleValue != parentState->admissibleValues.end();  admissibleValue += 1)
    {
        if (*admissibleValue != value and (not childState->isAdmissible(*admissibleValue)))
        {
            childState->admissibleValues.pushBack(admissibleValue);
        }
    };
    ifPickupAddDelivery(value, childState);
}

__host__ __device__
void DP::VRPModel::ifPickupAddDelivery(OP::ValueType value, VRPState* state) const
{
    OP::ValueType const * const pickup = thrust::find(thrust::seq, problem->pickups.begin(), problem->pickups.end(), value);
    if (pickup < problem->pickups.end())
    {
        unsigned int const deliveryIdx = problem->pickups.indexOf(pickup);
        OP::ValueType const delivery = *problem->deliveries[deliveryIdx];
        if(not state->isAdmissible(delivery))
        {
            state->admissibleValues.pushBack(&delivery);
        }
    }
}