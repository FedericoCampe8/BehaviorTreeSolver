#pragma once

#include "Model.cuh"
#include "VRPState.cuh"

namespace DP
{
    class VRPModel : public Model
    {
        public:
            using StateType = VRPState;

        public:
            VRPModel(OP::Problem const * problem);

            void makeRoot(State* root) const;
            __host__ __device__ void calcCosts(unsigned int variableIdx, State const * state, uint32_t* costs) const;
            __host__ __device__ void makeState(State const * parentState, unsigned int selectedValue, unsigned int childStateCost, State* childState) const;
            __host__ __device__ void mergeNextState(State const * parentState, unsigned int selectedValue, State* childState) const;

        private:
            __host__ __device__ void ifPickupAddDelivery(unsigned int selectedValue, VRPState* state) const;
    };
}