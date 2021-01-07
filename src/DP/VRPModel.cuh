#pragma once

#include "../OP/VRProblem.cuh"
#include "VRPState.cuh"

namespace DP
{
    void makeRoot(OP::VRProblem const * problem, VRPState* root);
    __global__ void calcCosts(OP::VRProblem const * problem, unsigned int variableIdx, Vector<VRPState>* currentStates, Vector<CostType>* costs);
    __global__ void makeStates(OP::VRProblem const * problem, unsigned int variableIdx, Vector<VRPState> const * currentStates, Vector<VRPState> const * nextStates, Vector<uint32_t> const * indices, Vector<CostType> const * costs);
}