#pragma once

#include "../DD/BuildMetadata.cuh"
#include "../OP/VRProblem.cuh"
#include "VRPState.cuh"

namespace DP
{
    void makeRoot(OP::VRProblem const * problem, VRPState* root);
    __global__ void calcCosts(OP::VRProblem const * problem, unsigned int variableIdx, Vector<VRPState>* currentStates, Vector<DD::BuildMetadata>* buildMetadata);
    __global__ void makeStates(OP::VRProblem const * problem, unsigned int variableIdx, Vector<VRPState> const * currentStates, Vector<VRPState> const * nextStates, Vector<DD::BuildMetadata>* buildMetadata);
}