#include <thrust/binary_search.h>
#include <thrust/equal.h>

#include "VRPModel.cuh"

DP::VRPModel::VRPModel(OP::VRProblem const * problem) :
    problem(problem)
{}


void DP::VRPModel::makeRoot(VRPState* root) const
{
    root->cost = 0;
    root->admissibleValues.incrementSize();
    *root->admissibleValues.back() = problem->start;
    root->admissibleValues.incrementSize();
    *root->admissibleValues.back() = problem->end;
    for(ValueType* pickup = problem->pickups.begin(); pickup != problem->pickups.end(); pickup += 1)
    {
        root->admissibleValues.incrementSize();
        *root->admissibleValues.back() = *pickup;
    }
}

__host__ __device__
void DP::VRPModel::calcCosts(unsigned int variableIdx, VRPState const * state, TS::Neighbourhood const * neighbourhood, uint32_t* costs) const
{
    using namespace OP;
    using namespace TS;

    OP::Variable const * const variable = problem->variables[variableIdx];

    for(Variable::ValueType* value = state->admissibleValues.begin(); value != state->admissibleValues.end(); value += 1)
    {
        bool const boundCheck = variable->minValue <= *value and *value <= variable->maxValue;
        bool isTabu = 0 < variableIdx and variableIdx < problem->variables.getCapacity() - 1;
        if(isTabu)
        {
            unsigned int fromVariable = variableIdx - 1;
            ValueType fromValue = *state->selectedValues.back();
            ValueType toValue = *value;
            Move move(fromVariable, fromValue, toValue);
            isTabu = neighbourhood->isTabu(&move);
        }

        if (boundCheck and (not isTabu))
        {
            unsigned int edgeIdx = *value - variable->minValue;
            costs[edgeIdx] = state->cost;
            if (not state->selectedValues.isEmpty())
            {
                costs[edgeIdx] += problem->getDistance(*state->selectedValues.back(), *value);
            }
        }
    }
}

__host__ __device__
void DP::VRPModel::makeState(VRPState const * parentState, ValueType selectedValue, unsigned int childStateCost, VRPState* childState) const
{
    // Initialize child state
    *childState = *parentState;
    childState->cost = childStateCost;

    // Remove value from admissible values
    assert(parentState->isAdmissible(selectedValue));
    childState->removeFromAdmissibles(selectedValue);

    // Add value to selected values
    childState->selectedValues.incrementSize();
    *childState->selectedValues.back() = static_cast<uint8_t>(selectedValue);

    ifPickupAddDelivery(selectedValue, childState);

    /*
    printf("[DEBUG] Selected: ");
    parentState->selectedValues.print(false);
    printf(" -%d-> ", selectedValue);
    childState->selectedValues.print(false);
    printf(" | Admissibles: ");
    parentState->admissibleValues.print(false);
    printf(" -%d-> ", selectedValue);
    childState->admissibleValues.print(true);
     */
}

__host__ __device__
void DP::VRPModel::mergeState(VRPState const * parentState, ValueType selectedValue, VRPState* childState) const
{
    // Merge admissible values
    for (ValueType* admissibleValue = parentState->admissibleValues.begin(); admissibleValue != parentState->admissibleValues.end();  admissibleValue += 1)
    {
        if (*admissibleValue != selectedValue and (not childState->isAdmissible(*admissibleValue)))
        {
            childState->admissibleValues.incrementSize();
            *childState->admissibleValues.back() = *admissibleValue;
        }
    };

    ifPickupAddDelivery(selectedValue, childState);
}

__host__ __device__
void DP::VRPModel::ifPickupAddDelivery(ValueType selectedValue, VRPState* state) const
{
    ValueType const * const pickup = thrust::find(thrust::seq, problem->pickups.begin(), problem->pickups.end(), selectedValue);
    if (pickup < problem->pickups.end())
    {
        unsigned int const deliveryIdx = problem->pickups.indexOf(pickup);
        ValueType const delivery = *problem->deliveries[deliveryIdx];
        if(not state->isAdmissible(delivery))
        {
            state->admissibleValues.incrementSize();
            *state->admissibleValues.back() = delivery;
        }
    }
}