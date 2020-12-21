#pragma once

#include "../OP/VRProblem.cuh"
#include "VRPState.cuh"

namespace DP
{
    class VRPModel
    {
        public:
            OP::VRProblem const & problem;

        public:
            VRPModel(OP::VRProblem const & problem);
            void makeRoot(VRPState& root) const;
            __host__ __device__ void calcCosts(unsigned int variableIdx, VRPState const & state, uint32_t* costs) const;
            __host__ __device__ void makeState(VRPState const & parentState, unsigned int selectedValue, unsigned int childStateCost, VRPState& childState) const;
            __host__ __device__ void mergeState(VRPState const & parentState, unsigned int selectedValue, VRPState& childState) const;

        private:
            __host__ __device__ void ifPickupAddDelivery(unsigned int selectedValue, VRPState& state) const;
    };
}