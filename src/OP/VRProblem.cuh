#pragma once

#include <cstddef>
#include <cstdint>
#include <Containers/Array.cuh>
#include <Containers/Vector.cuh>

#include "Problem.cuh"

namespace OP
{
    class VRProblem : public Problem
    {
        // Members
        public:
            ValueType start;
            ValueType end;
            Vector<ValueType> pickups;
            Vector<ValueType> deliveries;
            Array<ValueType> distances;

        // Functions
        public:
            VRProblem(unsigned int variablesCount, Memory::MallocType mallocType);
            __host__ __device__ unsigned int getDistance(ValueType from, ValueType to) const;
    };
}

