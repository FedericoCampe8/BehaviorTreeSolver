#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#include "TSPModel.cuh"

__host__
void DP::TSPModel::makeRoot(OP::TSPProblem const * problem, TSPState* root)
{
    root->cost = 0;
    root->addToAdmissibles(problem->startLocation);
    root->addToAdmissibles(problem->endLocation);
    thrust::for_each(thrust::seq, problem->pickups.begin(), problem->pickups.end(), [=] (auto& pickup)
    {
         root->addToAdmissibles(pickup);
    });
}

__device__
void DP::TSPModel::calcCosts(OP::TSPProblem const * problem, unsigned int level, TSPState const * state, int16_t* costs)
{
    OP::Variable const & var = problem->vars[level];
    thrust::for_each(thrust::seq, state->admissibleValues.begin(), state->admissibleValues.end(), [=] (auto& value)
    {
        if (var.minValue <= value and value <= var.maxValue)
        {
            unsigned int edgeIdx = value - problem->vars[level].minValue;
            costs[edgeIdx] = state->cost;
            if(not state->selectedValues.isEmpty())
            {
                costs[edgeIdx] += problem->getDistance(state->selectedValues.back(), value);
            }
        }
    });
}

__device__
void DP::TSPModel::makeNextState(OP::TSPProblem const * problem, TSPState const * state, int value, int cost, TSPState* nextState)
{
    *nextState = *state;
    nextState->cost = cost;
    if (not nextState->isSelected(value))
    {
        nextState->selectedValues.pushBack(value);
    }
    bool isPickup = thrust::binary_search(thrust::seq, problem->pickups.begin(), problem->pickups.end(), value);
    if (isPickup)
    {
        unsigned int* pickup = thrust::lower_bound(thrust::seq, problem->pickups.begin(), problem->pickups.end(), value);
        unsigned int deliveryIdx = thrust::distance(problem->pickups.begin(), pickup);
        unsigned int delivery = problem->deliveries[deliveryIdx];
        if(not nextState->isAdmissible(delivery))
        {
            nextState->addToAdmissibles(delivery);
        }
    }
    uint8_t* admissibleValuesBegin = nextState->admissibleValues.begin();
    uint8_t* admissibleValuesEnd = nextState->admissibleValues.end();
    admissibleValuesEnd = thrust::remove(thrust::seq, admissibleValuesBegin, admissibleValuesEnd, value);
    nextState->admissibleValues.resize(thrust::distance(admissibleValuesBegin, admissibleValuesEnd));
}

__device__
void DP::TSPModel::mergeNextState(OP::TSPProblem const * problem, TSPState const * state, int value, TSPState* nextState)
{
    // Merge admissible values
    thrust::for_each(thrust::seq, state->admissibleValues.begin(),  state->admissibleValues.end(), [=] (auto& admissibleValue)
    {
        if (value != admissibleValue and (not nextState->isAdmissible(admissibleValue)))
        {
            nextState->addToAdmissibles(admissibleValue);
        }
    });

    bool isPickup = thrust::binary_search(thrust::seq, problem->pickups.begin(), problem->pickups.end(), value);
    if (isPickup)
    {
        unsigned int* pickup = thrust::lower_bound(thrust::seq, problem->pickups.begin(), problem->pickups.end(), value);
        unsigned int deliveryIdx = thrust::distance(problem->pickups.begin(), pickup);
        unsigned int delivery = problem->deliveries[deliveryIdx];
        if(not nextState->isAdmissible(delivery))
        {
            nextState->addToAdmissibles(delivery);
        }
    }
}
