#pragma once

#include <cstdint>

namespace OP
{
    class Variable
    {
        public:
            uint8_t minValue;
            uint8_t maxValue;

        public:
            Variable(unsigned int minValue, unsigned int maxValue);
            static unsigned int cardinality(Variable const & variable);
    };
}