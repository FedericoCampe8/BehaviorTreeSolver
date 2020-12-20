#pragma once

#include "../OP/VRProblem.cuh"
#include "VRPState.cuh"

namespace DP
{
    class VRPModel
    {
        public:
            using StateType = DP::VRPState;
            using ProblemType = OP::VRProblem;

        public:
            ProblemType const & problem;

        public:
            VRPModel(ProblemType const & problem);
            void makeRoot(StateType& root) const;
            __host__ __device__ void calcCosts(unsigned int variableIdx, StateType const & state, uint32_t* costs) const;
            __host__ __device__ void makeState(StateType const & parentState, unsigned int selectedValue, unsigned int childStateCost, StateType& childState) const;
            __host__ __device__ void mergeState(StateType const & parentState, unsigned int selectedValue, StateType& childState) const;

        private:
            __host__ __device__ void ifPickupAddDelivery(unsigned int selectedValue, StateType& state) const;
    };
}