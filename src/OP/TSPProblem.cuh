#pragma once

#include <cstdint>

#include <Containers/StaticVector.cuh>
#include <Containers/RuntimeArray.cuh>

#include "Problem.cuh"

namespace OP
{
    class TSPProblem : public Problem
    {
        public:
            unsigned int startLocation;
            unsigned int endLocation;
            StaticVector<unsigned int> pickups;
            StaticVector<unsigned int> deliveries;
            RuntimeArray<unsigned int> distances;

        public:
            __host__ __device__ TSPProblem(unsigned int varsCount, std::byte* storage);
            __device__ TSPProblem& operator=(TSPProblem const & other);
            __host__ void setStartEndLocations(unsigned int startLocation, unsigned int endLocation);
            __host__ void addPickupDelivery(unsigned int pickup, unsigned int delivery);
            __device__ unsigned int const & getDistance(unsigned int from, unsigned int to) const;
            __host__ __device__ static std::size_t sizeOfStorage(unsigned int varsCount);

    };
}

