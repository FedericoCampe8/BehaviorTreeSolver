#pragma once

#include <cstdint>

namespace OP
{
    using ValueType = uint8_t;

    class Variable
    {
        // Members
        public:
            ValueType minValue;
            ValueType maxValue;

        // Functions
        public:
            Variable(ValueType minValue, ValueType maxValue);
    };
}