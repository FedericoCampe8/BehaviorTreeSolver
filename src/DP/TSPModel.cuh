#pragma once

#include "../OP/TSPProblem.cuh"
#include "TSPState.cuh"

namespace DP
{
    namespace TSPModel
    {
            __host__ void makeRoot(OP::TSPProblem const * problem,  TSPState* root);
            __device__ void makeNextState(OP::TSPProblem const * problem, TSPState const * state, int value, unsigned int cost, TSPState* nextState);
            __device__ void calcCosts(OP::TSPProblem const * problem, unsigned int level, TSPState const * state, uint32_t* costs);
    };
}