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
            bool active : 1;
            bool exact : 1;
            uint32_t cost : 30;
            StaticVector<int32_t> selectedValues;
            StaticVector<int32_t> admissibleValues;

        public:
            __host__ __device__ TSPState(OP::TSPProblem const * problem, std::byte* storage);
            __host__ __device__ TSPState& operator=(TSPState const & other);
            __host__ __device__ static void reset(TSPState& state);
            __host__ __device__ static std::size_t sizeofStorage(OP::TSPProblem const * problem);
            __host__ __device__ bool isAdmissible(int value) const;
            __host__ __device__ bool isSelected(int value) const;
            __host__ __device__ void addToAdmissibles(int value);

    };
}