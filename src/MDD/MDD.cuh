#pragma once

#include "DAG.cuh"
#include "../DP/TSPModel.cuh"
#include "../OP/TSPProblem.cuh"

namespace MDD
{
    class MDD
    {
        public:
            enum Type {Relaxed, Restricted};
            Type const type;
            DP::TSPState const * const root;
            unsigned int rootLvl;
            OP::TSPProblem const * const problem;
            DAG dag;

        public:
            __device__ MDD(Type type, unsigned int width, DP::TSPState const * const root, unsigned int rootLvl, OP::TSPProblem const * problem);
            __device__ void buildTopDown(std::byte * buffer);
            __device__ void print(unsigned int rootLvl = 0, bool endline = true) const;
        private:
            __device__ unsigned int calcFanout() const;
            __device__ unsigned int calcHeight() const;
            __device__ void copyEdgesToGlobal(Edge* edgesSrc, Edge* edgesDst, unsigned int count);
            __device__ void copyStatesToGlobal(DP::TSPState* stateSrc, std::byte* statesStorageSrc, DP::TSPState* stateDst, std::byte* statesStorageDst, unsigned int stateStorageSize, unsigned int count);
    };
}

