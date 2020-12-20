#pragma once

#include <cstddef>
#include <cstdint>
#include <Containers/Vector.cuh>

#include "../OP/Problem.cuh"

namespace DP
{
    class State
    {
        public:
            static const uint32_t MaxCost = UINT32_MAX;

        public:
            uint32_t cost;
            Vector<uint8_t> selectedValues;
            Vector<uint8_t> admissibleValues;

        public:
            // Storage
            __host__ __device__ static std::byte* mallocStorages(unsigned int statesCount, OP::Problem const & problem, Memory::MallocType mallocType);
            __host__ __device__ static std::size_t sizeOfStorage(OP::Problem const & problem);
            __host__ __device__ std::byte* storageEnd() const;

            // State
            __host__ __device__ State(OP::Problem const & problem, std::byte* storage);
            __host__ __device__ bool isAdmissible(unsigned int value) const;
            __host__ __device__ bool isSelected(unsigned int value) const;
            __host__ __device__ State& operator=(State const & other);
    };
}