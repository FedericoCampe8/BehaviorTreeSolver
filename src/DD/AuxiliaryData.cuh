#pragma once

#include <thrust/swap.h>
#include "../DP/Context.h"

namespace DD
{
    class AuxiliaryData
    {
        // Members
        public:
        DP::CostType cost;
        uint32_t index;

        // Functions
        public:
        __host__ __device__ AuxiliaryData();
        __host__ __device__ AuxiliaryData(DP::CostType cost, uint32_t index);
        __host__ __device__ AuxiliaryData& operator=(AuxiliaryData const & other);
        __host__ __device__ bool operator<(AuxiliaryData const & other) const;
        __host__ __device__ static void swap(AuxiliaryData& ad0, AuxiliaryData& ad1);
    };
}

__host__ __device__
DD::AuxiliaryData::AuxiliaryData() :
    AuxiliaryData(0, 0)
{}

__host__ __device__
DD::AuxiliaryData::AuxiliaryData(DP::CostType cost, uint32_t index) :
    cost(cost),
    index(index)
{}

__host__ __device__
DD::AuxiliaryData& DD::AuxiliaryData::operator=(DD::AuxiliaryData const & other)
{
    cost = other.cost;
    index = other.index;
    return *this;
}

__host__ __device__
bool DD::AuxiliaryData::operator<(DD::AuxiliaryData const & other) const
{
    return cost < other.cost;
}

__host__ __device__
void DD::AuxiliaryData::swap(DD::AuxiliaryData& ad0, DD::AuxiliaryData& ad1)
{
    thrust::swap(ad0.cost, ad1.cost);
    thrust::swap(ad0.index, ad1.index);
}