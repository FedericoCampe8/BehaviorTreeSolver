#include <thrust/for_each.h>
#include <thrust/binary_search.h>

#include "../OP/VRProblem.cuh"
#include "VRPState.cuh"
#include "VRPModel.cuh"

DP::VRPModel::VRPModel(OP::Problem const * problem) :
    Model(problem)
{}


void DP::VRPModel::makeRoot(State* root) const
{
    OP::VRProblem const * const vrProblem = reinterpret_cast<OP::VRProblem const *>(problem);
    VRPState* const vrRoot = reinterpret_cast<VRPState*>(root);

    vrRoot->cost = 0;
    thrust::for_each(thrust::seq, vrProblem->pickups.begin(), vrProblem->pickups.end(), [&] (uint8_t& pickup)
    {
         vrRoot->admissibleValues.pushBack(pickup);
    });
}

__host__ __device__
void DP::VRPModel::calcCosts(unsigned int variableIdx, State const * state, uint32_t* costs) const
{
    OP::VRProblem const * const vrProblem = reinterpret_cast<OP::VRProblem const *>(problem);
    VRPState const * const vrState = reinterpret_cast<VRPState const *>(state);

    OP::Variable const & variable = vrProblem->variables[variableIdx];
    for(uint8_t value = variable.minValue; value <= variable.maxValue; value += 1)
    {
        uint32_t& cost = costs[value - variable.minValue];
        if(vrState->isAdmissible(value))
        {
            cost = vrState->cost;
            if(not vrState->selectedValues.isEmpty())
            {
                costs += vrProblem->getDistance(vrState->selectedValues.back(), value);
            }
        }
        else
        {
            cost = State::MaxCost;
        }
    }
}

__host__ __device__
void DP::VRPModel::makeState(State const * parentState, unsigned int selectedValue, unsigned int childStateCost, State* childState) const
{
    VRPState const * const vrParentState = reinterpret_cast<VRPState const *>(parentState);
    VRPState* const vrChildState = reinterpret_cast<VRPState*>(childState);

    // Initialize child state
    *vrChildState = *vrParentState;
    vrChildState->cost = childStateCost;

    // Remove value from admissible values
    assert(vrChildState->isAdmissible(selectedValue));
    thrust::remove(thrust::seq, vrChildState->admissibleValues.begin(), vrChildState->admissibleValues.end(), static_cast<uint8_t>(selectedValue));
    vrChildState->admissibleValues.popBack();

    // Add value to selected values
    if (not vrChildState->isSelected(selectedValue))
    {
        vrChildState->selectedValues.pushBack(static_cast<uint8_t>(selectedValue));
    }

    ifPickupAddDelivery(selectedValue, vrChildState);
}

__host__ __device__
void DP::VRPModel::mergeNextState(State const * parentState, unsigned int selectedValue, State* childState) const
{
    VRPState const * const vrParentState = reinterpret_cast<VRPState const *>(parentState);
    VRPState* const vrChildState = reinterpret_cast<VRPState*>(childState);

    // Merge admissible values
    thrust::for_each(thrust::seq, vrParentState->admissibleValues.begin(), vrParentState->admissibleValues.end(), [&] (uint8_t& admissibleValue)
    {
        if (admissibleValue != selectedValue and (not vrChildState->isAdmissible(admissibleValue)))
        {
            vrChildState->admissibleValues.pushBack(admissibleValue);
        }
    });

    ifPickupAddDelivery(selectedValue, vrChildState);
}

__host__ __device__
void DP::VRPModel::ifPickupAddDelivery(unsigned int selectedValue, VRPState* state) const
{
    OP::VRProblem const * const vrProblem = reinterpret_cast<OP::VRProblem const *>(problem);

    uint8_t * const pickup = thrust::lower_bound(thrust::seq, vrProblem->pickups.begin(), vrProblem->pickups.end(), static_cast<uint8_t>(selectedValue));
    if (pickup != vrProblem->pickups.end())
    {
        unsigned int deliveryIdx = thrust::distance(vrProblem->pickups.begin(), pickup);
        uint8_t delivery = vrProblem->deliveries[deliveryIdx];
        if(not state->isAdmissible(delivery))
        {
            state->admissibleValues.pushBack(delivery);
        }
    }
}