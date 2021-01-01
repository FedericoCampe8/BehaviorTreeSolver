#pragma once

#include <cstdint>

namespace OP
{
    class Variable
    {
        // Aliases, Enums, ...
        public:
            using ValueType = uint8_t;

        // Members
        public:
            ValueType minValue;
            ValueType maxValue;

        // Functions
        public:
            Variable(unsigned int minValue, unsigned int maxValue);
            unsigned int cardinality();
    };
}