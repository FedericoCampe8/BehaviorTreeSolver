#pragma once

#include <thrust/swap.h>
#include "../DP/Context.h"

namespace DD
{
    class StateMetadata
    {
        // Members
        public:
        DP::CostType cost;
        u32 index;

        // Functions
        public:
        __host__ __device__ StateMetadata();
        __host__ __device__ StateMetadata(DP::CostType cost, u32 index);
        __host__ __device__ StateMetadata& operator=(StateMetadata const & other);
        __host__ __device__ bool operator<(StateMetadata const & other) const;
        __host__ __device__ bool operator==(StateMetadata const & other) const;
        __host__ __device__ static void swap(StateMetadata& sm0, StateMetadata& sm1);
        __host__ __device__ static bool isValid (StateMetadata const & sm);
    };
}

__host__ __device__
DD::StateMetadata::StateMetadata() :
    cost(0),
    index(0)
{}

__host__ __device__
DD::StateMetadata::StateMetadata(DP::CostType cost, u32 index) :
    cost(cost),
    index(index)
{}

__host__ __device__
DD::StateMetadata& DD::StateMetadata::operator=(DD::StateMetadata const & other)
{
    cost = other.cost;
    index = other.index;
    return *this;
}

__host__ __device__
bool DD::StateMetadata::operator<(DD::StateMetadata const & other) const
{
    return cost < other.cost;
}

__host__ __device__
void DD::StateMetadata::swap(DD::StateMetadata& sm0, DD::StateMetadata& sm1)
{
    thrust::swap(sm0.index, sm1.index);
    thrust::swap(sm0.cost, sm1.cost);
}

__host__ __device__
bool DD::StateMetadata::isValid(StateMetadata const & sm)
{
    return sm.cost != DP::MaxCost;
}

__host__ __device__
bool DD::StateMetadata::operator==(DD::StateMetadata const& other) const
{
    return cost == other.cost and index == other.index;
}