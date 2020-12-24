#pragma once

#include <cstdint>

namespace OP
{
    class Variable
    {
        // Members
        public:
            uint8_t minValue;
            uint8_t maxValue;

        // Functions
        public:
            Variable(unsigned int minValue, unsigned int maxValue);
            unsigned int cardinality() const;
            __host__ __device__ void print();
    };
}