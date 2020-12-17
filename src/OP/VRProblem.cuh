#pragma once

#include <cstddef>
#include <cstdint>

#include <Containers/RuntimeArray.cuh>
#include <Containers/StaticVector.cuh>

#include "Problem.cuh"

namespace OP
{
    class VRProblem : public Problem
    {
        public:
            uint8_t start;
            uint8_t end;
            std::byte padding[2]; // 32bit aligned
            StaticVector<uint8_t> pickups;
            StaticVector<uint8_t> deliveries;
            RuntimeArray<uint16_t> distances;

        public:
            VRProblem(unsigned int variablesCount, std::byte* storage);
            __host__ __device__ unsigned int getDistance(unsigned int from, unsigned int to) const;
            static std::size_t sizeOfStorage(unsigned int variablesCount);
    };
}

