#include <thrust/for_each.h>
#include <thrust/binary_search.h>

#include "VRPModel.cuh"

void DP::VRPModel::makeRoot(OP::VRP const & vrProblem, VRPState& root)
{
    root.cost = 0;
    thrust::for_each(thrust::seq, vrProblem.pickups.begin(), vrProblem.pickups.end(), [&] (OP::Variable::ValueType& pickup)
    {
         root.admissibleValues.pushBack(pickup);
    });
}

__host__ __device__
void DP::VRPModel::calcCosts(OP::VRP const & vrProblem, unsigned int variableIdx, VRPState const & state, VRPState::CostType* costs)
{
    OP::Variable const & variable = vrProblem.variables[variableIdx];
    for(OP::Variable::ValueType value = variable.minValue; value <= variable.maxValue; value += 1)
    {
        VRPState::CostType& cost = costs[value - variable.minValue];
        if(state.isAdmissible(value))
        {
            cost = state.cost;
            if(not state.selectedValues.isEmpty())
            {
                costs += vrProblem.getDistance(state.selectedValues.back(), value);
            }
        }
        else
        {
            cost = VRPState::MaxCost;
        }
    }
}

__host__ __device__
void DP::VRPModel::makeState(OP::VRP const & vrProblem, VRPState const & parentState, OP::Variable::ValueType selectedValue, VRPState::CostType childStateCost, VRPState& childState)
{
    // Initialize child state
    childState = parentState;
    childState.cost = childStateCost;

    // Remove value from admissible values
    assert(childState.isAdmissible(selectedValue));
    thrust::remove(thrust::seq, childState.admissibleValues.begin(), childState.admissibleValues.end(), selectedValue);
    childState.admissibleValues.popBack();

    // Add value to selected values
    if (not childState.isSelected(selectedValue))
    {
        childState.selectedValues.pushBack(selectedValue);
    }

    // If the value is a pickup, add the corresponding delivery
    OP::Variable::ValueType* pickup = thrust::lower_bound(thrust::seq, vrProblem.pickups.begin(), vrProblem.pickups.end(), value);
    if (pickup != vrProblem.pickups.end())
    {
        unsigned int deliveryIdx = thrust::distance(vrProblem.pickups.begin(), pickup);
        OP::Variable::ValueType delivery = vrProblem.deliveries[deliveryIdx];
        if(not childState.isAdmissible(delivery))
        {
            childState.admissibleValues.pushBack(delivery);
        }
    }
}

__host__ __device__
void DP::VRPModel::mergeNextState(OP::VRP const & vrProblem, VRPState const & parentState, OP::Variable::ValueType selectedValue, VRPState& childState)
{
    // Merge admissible values
    thrust::for_each(thrust::seq, parentState.admissibleValues.begin(), parentState.admissibleValues.end(), [=] (OP::Variable::ValueType& admissibleValue)
    {
        if (admissibleValue != selectedValue and (not childState.isAdmissible(admissibleValue)))
        {
            childState.admissibleValues.pushBack(admissibleValue);
        }
    });

    // If the value is a pickup, add the corresponding delivery
    OP::Variable::ValueType* pickup = thrust::lower_bound(thrust::seq, vrProblem.pickups.begin(), vrProblem.pickups.end(), value);
    if (pickup != vrProblem.pickups.end())
    {
        unsigned int deliveryIdx = thrust::distance(vrProblem.pickups.begin(), pickup);
        OP::Variable::ValueType delivery = vrProblem.deliveries[deliveryIdx];
        if(not childState.isAdmissible(delivery))
        {
            childState.admissibleValues.pushBack(delivery);
        }
    }
}
