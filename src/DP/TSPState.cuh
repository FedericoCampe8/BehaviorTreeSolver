#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <Containers/StaticVector.cuh>

#include "../OP/TSPProblem.cuh"
namespace DP
{
    class TSPState
    {
        public:
            int cost;
            StaticVector<unsigned int> selectedValues;
            StaticVector<unsigned int> admissibleValues;

        public:
            __host__ __device__ TSPState(OP::TSPProblem const * problem, std::byte* storage);
            __host__ __device__ TSPState& operator=(TSPState const & other);
            __host__ __device__ static void reset(TSPState& state);
            __host__ __device__ static std::size_t sizeOfStorage(OP::TSPProblem const * problem);
            __host__ __device__ bool isAdmissible(unsigned int value) const;
            __host__ __device__ bool isSelected(unsigned int value) const;
            __host__ __device__ void addToAdmissibles(unsigned int value);
    };
}