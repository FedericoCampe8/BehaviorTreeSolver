#pragma once

#include <cstddef>
#include <cstdint>

#include <Containers/StaticVector.cuh>

#include "../OP/VRP.cuh"

namespace DP
{
    class VRPState
    {

        public:
            using CostType = uint32_t;
            static CostType MaxCost = UINT32_MAX;

            CostType cost;
            StaticVector<OP::Variable::ValueType> selectedValues;
            StaticVector<OP::Variable::ValueType> admissibleValues;

        public:
            __host__ __device__ VRPState(unsigned int variablesCount, std::byte* storage);
            __host__ __device__ bool isAdmissible(OP::Variable::ValueType value) const;
            __host__ __device__ bool isSelected(OP::Variable::ValueType value) const;
            __host__ __device__ VRPState& operator=(VRPState const & other);
            __host__ __device__ static std::size_t sizeOfStorage(unsigned int variablesCount);
    };
}