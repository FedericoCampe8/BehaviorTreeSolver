#pragma once

#include "../OP/VRP.cuh"
#include "VRPState.cuh"

namespace DP
{
    class VRPModel
    {
        static void makeRoot(OP::VRP const & vrProblem, VRPState& root);
        __host__ __device__ static void calcCosts(OP::VRP const & vrProblem, unsigned int variableIdx, VRPState const & state, VRPState::CostType* costs);
        __host__ __device__ static void makeState(OP::VRP const & vrProblem, VRPState const & parentState, OP::Variable::ValueType selectedValue, VRPState::CostType childStateCost, VRPState& childState);
        __host__ __device__ static void mergeNextState(OP::VRP const & vrProblem, VRPState const & parentState, OP::Variable::ValueType selectedValue, VRPState& childState);
    };
}