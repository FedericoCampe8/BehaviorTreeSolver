#pragma once

#include <cstdint>

namespace OP
{
    class Variable
    {
        public:
            using ValueType = uint8_t;

            ValueType minValue;
            ValueType maxValue;

        public:
            Variable(ValueType minValue, ValueType maxValue);
            static unsigned int cardinality(Variable const & variable);
    };
}