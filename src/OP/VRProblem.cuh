#pragma once

#include <Containers/Array.cuh>
#include <Containers/Vector.cuh>
#include "../DP/Context.cuh"
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
        Array<DP::CostType> distances;

        // Functions
        public:
        __host__ __device__ unsigned int getDistance(ValueType from, ValueType to) const;
        static VRProblem* parseGrubHubInstance(char const * problemFileName, Memory::MallocType mallocType);
        protected:
        VRProblem(unsigned int variablesCount, Memory::MallocType mallocType);
    };
}

