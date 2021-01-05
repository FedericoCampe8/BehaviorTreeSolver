#pragma once

#include <cstddef>
#include <cstdint>
#include <Containers/LightVector.cuh>

#include "../OP/Problem.cuh"

namespace DP
{
    class State
    {
        // Aliases, Enums, ...
        public:
            enum : uint32_t {MaxCost = UINT32_MAX};

        // Members
        public:
            uint32_t cost;
            LightVector<uint8_t> selectedValues;
            LightVector<uint8_t> admissibleValues;

        // Functions
        public:
            __host__ __device__ State(OP::Problem const * problem, std::byte* storage);
            __host__ __device__ bool isAdmissible(unsigned int value) const;
            __host__ __device__ bool isSelected(unsigned int value) const;
            __host__ __device__ static std::byte* mallocStorages(OP::Problem const * problem, unsigned int statesCount, Memory::MallocType mallocType);
            __host__ __device__ void operator=(State const & other);
            __host__ __device__ void removeFromAdmissibles(unsigned int value);
            __host__ __device__ void reset();
            __host__ __device__ static std::size_t sizeOfStorage(OP::Problem const * problem);
            __host__ __device__ void print() const;
    };
}