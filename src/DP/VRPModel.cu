#include <thrust/for_each.h>
#include <thrust/binary_search.h>

#include "VRPModel.cuh"

DP::VRPModel::VRPModel(ProblemType const & problem) :
    problem(problem)
{}


void DP::VRPModel::makeRoot(StateType& root) const
{
    root.cost = 0;
    thrust::for_each(thrust::seq, problem.pickups.begin(), problem.pickups.end(), [&] (uint8_t& pickup)
    {
         root.admissibleValues.pushBack(pickup);
    });
}

__host__ __device__
void DP::VRPModel::calcCosts(unsigned int variableIdx, StateType const & state, uint32_t* costs) const
{
    OP::Variable const & variable = problem.variables[variableIdx];
    for(uint8_t value = variable.minValue; value <= variable.maxValue; value += 1)
    {
        uint32_t& cost = costs[value - variable.minValue];
        if(state.isAdmissible(value))
        {
            cost = state.cost;
            if(not state.selectedValues.isEmpty())
            {
                costs += problem.getDistance(state.selectedValues.back(), value);
            }
        }
        else
        {
            cost = StateType::MaxCost;
        }
    }
}

__host__ __device__
void DP::VRPModel::makeState(StateType const & parentState, unsigned int selectedValue, unsigned int childStateCost, StateType& childState) const
{
    // Initialize child state
    childState = parentState;
    childState.cost = childStateCost;

    // Remove value from admissible values
    assert(childState.isAdmissible(selectedValue));
    thrust::remove(thrust::seq, childState.admissibleValues.begin(), childState.admissibleValues.end(), static_cast<uint8_t>(selectedValue));
    childState.admissibleValues.popBack();

    // Add value to selected values
    if (not childState.isSelected(selectedValue))
    {
        childState.selectedValues.pushBack(static_cast<uint8_t>(selectedValue));
    }

    ifPickupAddDelivery(selectedValue, childState);
}

__host__ __device__
void DP::VRPModel::mergeState(StateType const & parentState, unsigned int selectedValue, StateType& childState) const
{
    // Merge admissible values
    thrust::for_each(thrust::seq, parentState.admissibleValues.begin(), parentState.admissibleValues.end(), [&] (uint8_t& admissibleValue)
    {
        if (admissibleValue != selectedValue and (not childState.isAdmissible(admissibleValue)))
        {
            childState.admissibleValues.pushBack(admissibleValue);
        }
    });

    ifPickupAddDelivery(selectedValue, childState);
}

__host__ __device__
void DP::VRPModel::ifPickupAddDelivery(unsigned int selectedValue, StateType& state) const
{
    uint8_t * const pickup = thrust::lower_bound(thrust::seq, problem.pickups.begin(), problem.pickups.end(), static_cast<uint8_t>(selectedValue));
    if (pickup != problem.pickups.end())
    {
        unsigned int deliveryIdx = static_cast<unsigned int>(thrust::distance(problem.pickups.begin(), pickup));
        uint8_t delivery = problem.deliveries[deliveryIdx];
        if(not state.isAdmissible(delivery))
        {
            state.admissibleValues.pushBack(delivery);
        }
    }
}