#pragma once

#include "../OP/TSPProblem.cuh"
#include "TSPState.cuh"

namespace DP
{
    namespace TSPModel
    {
            __host__ void makeRoot(OP::TSPProblem const * problem,  TSPState* root);
            __device__ void makeNextState(OP::TSPProblem const * problem, TSPState const * state, int value, int cost, TSPState* nextState);
            __device__ void mergeNextState(OP::TSPProblem const * problem,  TSPState const * state, int value, TSPState* nextState);
            __device__ void calcCosts(OP::TSPProblem const * problem, unsigned int level, TSPState const * state, int16_t * costs);
    };
}