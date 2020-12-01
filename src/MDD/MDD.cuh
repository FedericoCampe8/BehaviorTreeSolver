#pragma once

#include "DAG.cuh"
#include "../DP/TSPModel.cuh"
#include "../OP/TSPProblem.cuh"

namespace MDD
{
    class MDD
    {
        public:
            enum Type
            {
                Relaxed, Restricted
            };
            Type type;
            DP::TSPState const * const root;
            unsigned int rootLvl;
            OP::TSPProblem const * const problem;
            DAG dag;
            DP::TSPState * const cutset;


        public:
            __device__ MDD(Type type, unsigned int width, DP::TSPState const * const root, unsigned int rootLvl, OP::TSPProblem const* problem, DP::TSPState * const cutset);
            __device__ void buildTopDown(std::byte* nextState);
            __device__ unsigned int getMinCost() const;
            __device__ void print(unsigned int rootLvl = 0, bool endline = true) const;
            __host__ __device__ static unsigned int calcFanout(OP::TSPProblem const* problem, unsigned int rootLvl);
        private:
            __device__ unsigned int calcHeight() const;
    };
}