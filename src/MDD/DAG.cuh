#pragma once

#include <cstdint>

#include <Containers/RuntimeArray.cuh>

#include "Edge.cuh"
#include "../DP/TSPState.cuh"

namespace MDD
{
    class DAG
    {
        public:
            unsigned int const width;
            unsigned int const fanout;
            unsigned int const height;
            unsigned int const stateStorageSize;
        private:
            RuntimeArray<Edge> edges;
            RuntimeArray<DP::TSPState> states;
            RuntimeArray<std::byte> statesStorage;

        public:
            __device__ DAG(unsigned int width, unsigned int fanout, unsigned int height);
            __device__ Edge& getEdge(unsigned int level, unsigned int nodeIdx, unsigned int edgeIdx) const;
            __device__ Edge* getEdges(unsigned int level) const;
            __device__ DP::TSPState* getStates(unsigned int level) const;
            __device__ std::byte* getStatesStorage(unsigned int level) const;

        private:
            __device__ std::byte* mallocStatesStorage() const;
    };
}

