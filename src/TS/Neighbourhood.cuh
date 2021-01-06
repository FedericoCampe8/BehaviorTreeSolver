#pragma once

#include <Containers/CircularBuffer.cuh>

#include "../OP/Problem.cuh"
#include "Move.cuh"

namespace TS
{
    class Neighbourhood
    {
        // Aliases, Enums, ...
        public:
            using ValueType = OP::Variable::ValueType;

        // Members
        private:
            unsigned int bestAvgCost;
            unsigned int updatesCount;
            CircularBuffer<Move> shortTermTabuMoves;
            CircularBuffer<Move> midTermTabuMoves;
            CircularBuffer<Move> longTermTabuMoves;

        // Functions
        public:
            Neighbourhood(OP::Problem const* problem, unsigned int tabuListSize, Memory::MallocType mallocType);
            __host__ __device__ bool isTabu(Move const * move) const;
            void update(LightVector<OP::Variable::ValueType> const * solution);
    };
}