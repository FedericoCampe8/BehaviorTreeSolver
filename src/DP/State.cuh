#pragma once

#include <cstddef>
#include <cstdint>

#include <Containers/StaticVector.cuh>

#include "../OP/Problem.cuh"

namespace DP
{
    class State
    {
        public:
            static const uint32_t MaxCost = UINT32_MAX;

        public:
            uint32_t cost;
            StaticVector<uint8_t> selectedValues;
            StaticVector<uint8_t> admissibleValues;

        public:
            __host__ __device__ State(unsigned int variablesCount, std::byte* storage);
            __host__ __device__ bool isAdmissible(unsigned int value) const;
            __host__ __device__ bool isSelected(unsigned int value) const;
            __host__ __device__ State& operator=(State const & other);
            __host__ __device__ static std::size_t sizeOfStorage(unsigned int variablesCount);
    };
}