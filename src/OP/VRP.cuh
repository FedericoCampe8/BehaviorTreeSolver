#pragma once

#include <cstdint>

#include <Containers/StaticVector.cuh>
#include <Containers/RuntimeArray.cuh>

#include "Variable.cuh"

namespace OP
{
    class VRP
    {
        public:
            using DistanceType = uint16_t;

            RuntimeArray<Variable> variables;
            StaticVector<Variable::ValueType> pickups;
            StaticVector<Variable::ValueType> deliveries;
            RuntimeArray<DistanceType> distances;

        public:
            VRP(unsigned int variablesCount, std::byte* storage);
            void addPickupDelivery(Variable::ValueType pickup, Variable::ValueType delivery);
            __host__ __device__ DistanceType getDistance(Variable::ValueType from, Variable::ValueType to) const;
            static std::size_t sizeOfStorage(unsigned int variablesCount);

    };
}

