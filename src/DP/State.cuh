#pragma once

#include <cstddef>
#include <cstdint>
#include <Containers/LightVector.cuh>

#include "../OP/Problem.cuh"

namespace DP
{
    using CostType = uint32_t;
    enum {MaxCost = UINT32_MAX}; // GPU workaround for const values

    class State
    {
        // Members
        public:
            CostType cost;
            LightVector<OP::ValueType> selectedValues;
            LightVector<OP::ValueType> admissibleValues;

        // Functions
        public:
            __host__ __device__ State(OP::Problem const * problem, std::byte* storage);
            __host__ __device__ bool isAdmissible(OP::ValueType value) const;
            __host__ __device__ bool isSelected(OP::ValueType value) const;
            __host__ __device__ static std::byte* mallocStorages(OP::Problem const * problem, unsigned int statesCount, Memory::MallocType mallocType);
            __host__ __device__ void operator=(State const & other);
            __host__ __device__ void removeFromAdmissibles(OP::ValueType value);
            __host__ __device__ void reset();
            __host__ __device__ static unsigned int sizeOfStorage(OP::Problem const * problem);
            __host__ __device__ void print(bool endLine = true) const;
    };
}