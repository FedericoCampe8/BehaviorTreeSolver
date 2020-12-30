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
            void fixTo(unsigned int value);
            bool isFixed() const;
            unsigned int cardinality() const;
    };
}