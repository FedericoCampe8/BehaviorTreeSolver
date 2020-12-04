#pragma once

#include "../DP/TSPModel.cuh"
#include "../OP/TSPProblem.cuh"

namespace MDD
{
    class MDD
    {
        public:
            enum Type { Relaxed, Restricted };
            Type type;
            unsigned int const width;
            unsigned int const fanout;
            DP::TSPState const * const top;
            OP::TSPProblem const * const problem;

        public:
            __device__ MDD(Type type, unsigned int width, DP::TSPState const * top, OP::TSPProblem const * problem);
            __device__ void buildTopDown(DP::TSPState* bottom, unsigned int& cutsetSize, DP::TSPState * const cutset, std::byte* buffer);

            __host__ __device__ static unsigned int calcFanout(OP::TSPProblem const* problem);
    };
}