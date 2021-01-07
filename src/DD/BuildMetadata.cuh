#pragma once

#include <cstdint>
#include "../DP/Context.cuh"

namespace DD
{
    class BuildMetadata
    {
        // Members
        public:
        DP::CostType cost;
        uint32_t index;

        // Functions
        public:
        __host__ __device__ inline BuildMetadata();
        __host__ __device__ inline BuildMetadata(DP::CostType cost, uint32_t index);
        __host__ __device__ inline bool operator<(BuildMetadata const& other) const;
        __host__ __device__ inline BuildMetadata& operator=(BuildMetadata const& other);
    };

    __host__ __device__
    BuildMetadata::BuildMetadata() :
        BuildMetadata(DP::MaxCost, 0)
    {}

    __host__ __device__
    BuildMetadata::BuildMetadata(DP::CostType cost, uint32_t index) :
        cost(cost),
        index(index)
    {}

    __host__ __device__
    bool BuildMetadata::operator<(BuildMetadata const & other) const
    {
        return cost < other.cost;
    }

    __host__ __device__
    BuildMetadata& BuildMetadata::operator=(BuildMetadata const & other)
    {
        cost = other.cost;
        index = other.index;
        return *this;
    }
}

