#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#include "../OP/TSPProblem.cuh"
#include "TSPModel.cuh"

__host__
void DP::TSPModel::makeRoot(OP::TSPProblem const * problem,  TSPState* root)
{
    root->type = TSPState::Type::Active;
    root->cost = 0;
    root->lastValue = problem->startLocation;
    root->addToAdmissibles(problem->startLocation);
    root->addToAdmissibles(problem->endLocation);
    thrust::for_each(thrust::seq, problem->pickups.begin(), problem->pickups.end(), [=] (auto& pickup)
    {
         root->addToAdmissibles(pickup);
    });
}

__device__
void DP::TSPModel::makeNextState(OP::TSPProblem const * problem, TSPState const * state, int value, unsigned int cost, TSPState* nextState)
{
    *nextState = *state;
    nextState->cost = cost;
    nextState->lastValue = value;
    bool isPickup = thrust::binary_search(thrust::seq, problem->pickups.begin(), problem->pickups.end(), value);
    if(isPickup)
    {
        uint16_t* pickup = thrust::lower_bound(thrust::seq, problem->pickups.begin(), problem->pickups.end(), value);
        uint16_t deliveryIdx = thrust::distance(problem->pickups.begin(), pickup);
        uint16_t delivery = problem->deliveries[deliveryIdx];
        nextState->addToAdmissibles(delivery);
        nextState->admissibleValues.remove(value);
    }
    else
    {
        nextState->admissibleValues.remove(value);
    }
}

__device__
void DP::TSPModel::calcCosts(OP::TSPProblem const * problem, unsigned int level, TSPState const * state, uint32_t* costs)
{
    OP::Variable const & var = problem->vars[level];
    thrust::for_each(thrust::seq, state->admissibleValues.begin(), state->admissibleValues.end(), [=] (auto& value)
    {
        if(var.minValue <= value and value <= var.maxValue)
        {
            unsigned int edgeIdx = value - problem->vars[level].minValue;
            costs[edgeIdx] = state->cost + problem->getDistance(state->lastValue, value);
        }
    });
}
