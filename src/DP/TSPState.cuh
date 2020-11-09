#pragma once

#include <cstddef>
#include <cstdint>

#include <Containers/StaticVector.cuh>

namespace DP
{
    class TSPState
    {
        public:
            enum Type: uint32_t {Active, NotActive};
            Type type: 1;
            uint32_t cost;
            int32_t lastValue;
            StaticVector<int32_t> admissibleValues;

        public:
            __host__ __device__ TSPState(unsigned int height, std::byte* storage);
            __device__ TSPState& operator=(TSPState const & other);
            __device__ static void reset(TSPState& state);
            __device__ static bool isActive(TSPState const & state);
            __host__ __device__ static std::size_t sizeofStorage(unsigned int capacity);
            __host__ __device__ bool isAdmissible(int value) const;
            __host__ __device__ void addToAdmissibles(int value);

    };
}