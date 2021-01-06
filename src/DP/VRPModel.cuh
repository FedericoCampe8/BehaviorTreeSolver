#pragma once

#include "../OP/VRProblem.cuh"
#include "VRPState.cuh"
#include "../LNS/Neighbourhood.cuh"

namespace DP
{
    class VRPModel
    {
        // Members
        public:
            OP::VRProblem const * const problem;

        // Functions
        public:
            VRPModel(OP::VRProblem const * problem);
            void makeRoot(VRPState* root) const;
            __host__ __device__ void calcCosts(unsigned int variableIdx, VRPState const * state, LNS::Neighbourhood const * neighbourhood, CostType* costs) const;
            __host__ __device__ void makeState(VRPState const * parentState, OP::ValueType value, uint32_t childStateCost, VRPState* childState) const;
            __host__ __device__ void mergeState(VRPState const * parentState, OP::ValueType value, VRPState* childState) const;
        private:
            __host__ __device__ void ifPickupAddDelivery(OP::ValueType value, VRPState* state) const;
    };
}