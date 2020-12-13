#pragma once

#include "../DP/TSPModel.cuh"
#include "../OP/TSPProblem.cuh"

namespace MDD
{
    enum MDDType {Relaxed, Restricted};
    __device__ void buildMddTopDown(OP::TSPProblem const* problem, unsigned int maxWidth, MDDType type, DP::TSPState& top, unsigned int cutsetMaxSize, unsigned int& cutsetSize, DP::TSPState* cutset, DP::TSPState& bottom, std::byte* scratchpad);
    __host__ __device__ unsigned int calcFanout(OP::TSPProblem const* problem);
}