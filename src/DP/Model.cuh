#pragma once

#include "../OP/Problem.cuh"
#include "State.cuh"

namespace DP
{
    class Model
    {
        public:
            using StateType = State;

        public:
            OP::Problem const * problem;

        public:
            Model(OP::Problem const * problem);

            virtual void makeRoot(State* root) const = 0;
            __host__ __device__ virtual void calcCosts(unsigned int variableIdx, State const * state, uint32_t* costs) const = 0 ;
            __host__ __device__ virtual void makeState(State const * parentState, unsigned int selectedValue, unsigned int childStateCost, State* childState) const = 0;
            __host__ __device__ virtual void mergeNextState(State const * parentState, unsigned int selectedValue, State* childState) const = 0;
    };
}