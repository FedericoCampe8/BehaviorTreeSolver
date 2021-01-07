#pragma once

#include <cstdint>
#include "Context.cuh"

namespace OP
{
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