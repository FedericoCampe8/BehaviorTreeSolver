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
            uint16_t startLocation;
            uint16_t endLocation;
            StaticVector<uint16_t> pickups;
            StaticVector<uint16_t> deliveries;
            RuntimeArray<uint16_t> distances;

        public:
            __host__ __device__ TSPProblem(unsigned int varsCount, std::byte* storage);
            __device__ TSPProblem& operator=(TSPProblem const & other);
            __host__ void setStartEndLocations(unsigned int startLocation, unsigned int endLocation);
            __host__ void addPickupDelivery(unsigned int pickup, unsigned int delivery);
            __device__ uint16_t const & getDistance(unsigned int from, unsigned int to) const;
            __host__ __device__ static std::size_t sizeofStorage(unsigned int varsCount);

    };
}

