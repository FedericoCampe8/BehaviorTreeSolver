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
            uint8_t start;
            uint8_t end;
            std::byte padding[6]; // 64bit aligned
            Vector<uint8_t> pickups;
            Vector<uint8_t> deliveries;
            Array<uint16_t> distances;

        // Functions
        public:
            VRProblem(unsigned int variablesCount, Memory::MallocType mallocType);
            __host__ __device__ unsigned int getDistance(unsigned int from, unsigned int to) const;
    };
}

